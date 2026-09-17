// Exact best responses for the two-player clique allocation game.
#include <algorithm>
#include <omp.h>
#include <chrono>
#include <queue>
#include <array>
#include <set>
#include <cmath>
#include <cstring>
#include <vector>

namespace {

const double NEG = -1e18;

struct Entry { int size, a, b, mult; };

double optimality_scale(const std::vector<Entry>& live, int total, int omega,
                        double difficulty, std::vector<int>& sizes,
                        std::vector<double>& opt) {
  sizes.clear();
  for (const Entry& e : live) {
    if (std::find(sizes.begin(), sizes.end(), e.size) == sizes.end())
      sizes.push_back(e.size);
  }
  std::sort(sizes.begin(), sizes.end(), std::greater<int>());
  opt.assign(sizes.size(), 0.0);
  double inv = 1.0 / static_cast<double>(total);
  int run = 0;
  double top = 0.0;
  for (size_t i = 0; i < sizes.size(); ++i) {
    double rel = static_cast<double>(sizes[i]) / static_cast<double>(omega);
    opt[i] = std::exp(-(run * inv) / rel);
    if (opt[i] > top) top = opt[i];
    for (const Entry& e : live)
      if (e.size == sizes[i]) run += (e.a + e.b) * e.mult;
  }
  return (1.0 + difficulty) / top;
}

void score_board(const std::vector<Entry>& board, double difficulty,
                 double* mean_a, double* mean_b) {
  // Hot path: called ~2.1M times per solve. The old version built a `live`
  // vector, then used std::find + std::sort + two more vectors to handle what is
  // only ever a couple of distinct clique sizes. Same arithmetic, same order,
  // no containers: distinct sizes go in a fixed stack array.
  enum { MAXS = 8 };
  int sizes[MAXS];
  long long held_sum[MAXS];
  int ns = 0;
  int total = 0, omega = 0, x_min = 1 << 30;
  bool any = false;
  for (const Entry& e : board) {
    int held = e.a + e.b;
    if (e.mult <= 0 || held <= 0) continue;
    any = true;
    total += held * e.mult;
    if (e.size > omega) omega = e.size;
    if (held < x_min) x_min = held;
    int k = -1;
    for (int j = 0; j < ns; ++j) if (sizes[j] == e.size) { k = j; break; }
    if (k < 0) {
      if (ns == MAXS) continue;            // cannot happen for omega/omega-1
      k = ns++; sizes[k] = e.size; held_sum[k] = 0;
    }
    held_sum[k] += static_cast<long long>(held) * e.mult;
  }
  if (!any) { *mean_a = *mean_b = 0.0; return; }
  // descending by size, as std::sort(greater<int>) gave
  for (int i = 1; i < ns; ++i)
    for (int j = i; j > 0 && sizes[j] > sizes[j - 1]; --j) {
      int ts = sizes[j]; sizes[j] = sizes[j - 1]; sizes[j - 1] = ts;
      long long th = held_sum[j]; held_sum[j] = held_sum[j - 1]; held_sum[j - 1] = th;
    }
  double opt[MAXS];
  double inv = 1.0 / static_cast<double>(total);
  long long run = 0;
  double top = 0.0;
  for (int i = 0; i < ns; ++i) {
    double rel = static_cast<double>(sizes[i]) / static_cast<double>(omega);
    opt[i] = std::exp(-(run * inv) / rel);
    if (opt[i] > top) top = opt[i];
    run += held_sum[i];
  }
  double scale = (1.0 + difficulty) / top;
  double sa = 0.0, sb = 0.0;
  long ca = 0, cb = 0;
  for (const Entry& e : board) {
    int held = e.a + e.b;
    if (e.mult <= 0 || held <= 0) continue;
    int k = 0;
    for (int j = 0; j < ns; ++j) if (sizes[j] == e.size) { k = j; break; }
    double r = opt[k] * scale + x_min / static_cast<double>(held);
    sa += e.a * e.mult * r; ca += e.a * e.mult;
    sb += e.b * e.mult * r; cb += e.b * e.mult;
  }
  *mean_a = ca ? sa / ca : 0.0;
  *mean_b = cb ? sb / cb : 0.0;
}


struct Item { int kind, index, low, high; };   // kind 0 occupied, 1 fresh

double gain_of(const std::vector<Entry>& board, const Item& it, int take,
               int xmin, int q, int held_a, double w_a = 1.0,
               double w_b = 1.0) {
  if (it.kind == 1)
    return take ? w_b * xmin / static_cast<double>(q) : 0.0;  // b == take
  const Entry& e = board[it.index];
  int total = e.a + e.b + take;
  // B's count on this clique is its EXISTING hotkeys plus the new ones. Using
  // `take` alone drops e.b from the numerator -- invisible on a board the first
  // player has just committed (e.b == 0), wrong everywhere else.
  return w_b * xmin * (e.b + take) / static_cast<double>(total)
             / static_cast<double>(q)
       - w_a * xmin * e.a / static_cast<double>(total)
             / static_cast<double>(held_a);
}

bool allocate(const std::vector<Entry>& board, const std::vector<int>& occupied,
              int freeCliques, int budget, int xmin, int q_total, int held_a,
              int anchor, std::vector<int>* add, std::vector<int>* fresh,
              double w_a = 1.0, double w_b = 1.0) {
  add->assign(occupied.size(), 0);
  fresh->clear();
  if (budget < 0) return false;
  int n_fresh_max = xmin > 0 ? std::min(freeCliques, budget / xmin) : 0;

  double best_total = -1e300;
  std::vector<int> best_add;
  int best_fresh = -1;
  int best_j = 0;

  // A fresh clique is LUMPY: it costs xmin hotkeys at once, so its gain is a
  // staircase and marginal greedy does not apply to it. Enumerate how many to
  // open, and run the greedy -- exact for the concave occupied terms -- on the
  // budget that is left.
  // anchor == -2 means a newly opened clique attains xmin. It can only live in
  // ONE size class, so a class with no free capacity relaxes to no constraint
  // rather than killing the whole slice.
  int fresh_lo = (anchor == -2 && n_fresh_max >= 1) ? 1 : 0;

  // Every use_fresh iteration restarts take[] from the SAME low[] and differs
  // only in how many units are left to place, so the greedy pick sequence for a
  // smaller budget is a PREFIX of the sequence for the largest one. Run the
  // greedy once at the largest budget, record the picks and the running total,
  // then read every iteration off as a prefix. That removes the ~7x repetition
  // measured at 51.2M use_fresh iterations over 7.2M allocate calls.
  static thread_local std::vector<int> low, picks;
  static thread_local std::vector<double> cum;
  low.assign(occupied.size(), 0);
  int sum_low = 0;
  for (size_t k = 0; k < occupied.size(); ++k) {
    int i = occupied[k];
    int v = std::max(0, xmin - board[i].a - board[i].b);
    low[k] = v;
    sum_low += v;
  }
  int base_left = budget - sum_low;                 // before any fresh cliques
  int max_picks = base_left - fresh_lo * xmin;
  if (max_picks < 0) max_picks = -1;                // no feasible use_fresh

  picks.clear();
  cum.clear();
  double base_gain = 0.0;
  if (max_picks >= 0) {
    static thread_local std::vector<int> cur_take;
    cur_take = low;
    for (size_t k = 0; k < occupied.size(); ++k)
      base_gain += gain_of(board, Item{0, occupied[k], 0, budget}, cur_take[k],
                           xmin, q_total, held_a, w_a, w_b);
    cum.push_back(base_gain);
    struct Cand { double g; double cur; int k; };
    struct CandLess {
      bool operator()(const Cand& x, const Cand& y) const {
        if (x.g != y.g) return x.g < y.g;
        return x.k > y.k;                           // ties -> lowest index
      }
    };
    auto gain_at = [&](int k, int t) {
      return gain_of(board, Item{0, occupied[k], 0, budget}, t, xmin,
                     q_total, held_a, w_a, w_b);
    };
    if (max_picks > 0 && !occupied.empty()) {
      static thread_local std::vector<Cand> heap_buf;
      heap_buf.clear();
      for (size_t k = 0; k < occupied.size(); ++k) {
        if (occupied[k] == anchor) continue;        // anchor pinned at xmin
        int kk = static_cast<int>(k);
        double c0 = gain_at(kk, cur_take[kk]);
        heap_buf.push_back(Cand{gain_at(kk, cur_take[kk] + 1) - c0, c0, kk});
      }
      std::priority_queue<Cand, std::vector<Cand>, CandLess> pq(
          CandLess(), std::move(heap_buf));
      double running = base_gain;
      int placed = 0;
      while (placed < max_picks && !pq.empty()) {
        Cand top = pq.top();
        pq.pop();
        cur_take[top.k] += 1;
        running += top.g;
        picks.push_back(top.k);
        cum.push_back(running);
        ++placed;
        double c = top.cur + top.g;
        pq.push(Cand{gain_at(top.k, cur_take[top.k] + 1) - c, c, top.k});
      }
      heap_buf.clear();
    }
  }

  for (int use_fresh = fresh_lo; use_fresh <= n_fresh_max; ++use_fresh) {
    int left = budget - use_fresh * xmin;
    if (left < 0) break;
    left -= sum_low;
    if (left < 0) continue;
    if (left > 0 && use_fresh == 0 && occupied.empty()) continue;
    size_t j = static_cast<size_t>(left);
    if (j >= cum.size()) j = cum.empty() ? 0 : cum.size() - 1;
    double total = use_fresh * w_b * (static_cast<double>(xmin) / q_total)
                 + (cum.empty() ? 0.0 : cum[j]);
    if (total > best_total) {
      best_total = total; best_fresh = use_fresh; best_j = static_cast<int>(j);
    }
  }
  if (best_fresh >= 0) {
    best_add = low;
    for (int i = 0; i < best_j; ++i) best_add[picks[i]] += 1;
  }
  if (best_fresh < 0) return budget == 0;
  *add = best_add;
  int spent = best_fresh * xmin;
  for (int v : best_add) spent += v;
  for (int j = 0; j < best_fresh; ++j) fresh->push_back(xmin);
  if (spent < budget) {
    if (best_fresh > 0) fresh->back() += budget - spent;
    else if (!add->empty()) (*add)[0] += budget - spent;
    else return false;
  }
  return true;
}


double solve_core(std::vector<Entry>& board, int q, int omega, int n_top,
                  int n_spare, double difficulty,
                  std::vector<Entry>* out_board, double* out_a, double* out_b,
                  double w_a = 1.0, double w_b = 1.0);

}  // namespace

extern "C" {

void bs_score(const int* flat, int n, double difficulty, double* out) {
  std::vector<Entry> board(n);
  for (int i = 0; i < n; ++i)
    board[i] = Entry{flat[4 * i], flat[4 * i + 1], flat[4 * i + 2],
                     flat[4 * i + 3]};
  score_board(board, difficulty, out, out + 1);
}

double bs_best_response_w(const int* flat, int n, int q, int omega, int n_top,
                          int n_spare, double difficulty, double w_a,
                          double w_b, int* out_board, int* out_n,
                          double* out_means) {
  std::vector<Entry> board(n);
  for (int i = 0; i < n; ++i)
    board[i] = Entry{flat[3 * i], flat[3 * i + 1], flat[3 * i + 2], 1};
  std::vector<Entry> best;
  double gap = solve_core(board, q, omega, n_top, n_spare, difficulty, &best,
                          out_means, out_means + 1, w_a, w_b);
  *out_n = static_cast<int>(best.size());
  for (size_t i = 0; i < best.size(); ++i) {
    out_board[3 * i] = best[i].size;
    out_board[3 * i + 1] = best[i].a;
    out_board[3 * i + 2] = best[i].b;
  }
  return gap;
}

double bs_best_response(const int* flat, int n, int q, int omega, int n_top,
                        int n_spare, double difficulty, int* out_board,
                        int* out_n, double* out_means) {
  std::vector<Entry> board(n);
  for (int i = 0; i < n; ++i)
    board[i] = Entry{flat[3 * i], flat[3 * i + 1], flat[3 * i + 2], 1};
  std::vector<Entry> best;
  double gap = solve_core(board, q, omega, n_top, n_spare, difficulty, &best,
                          out_means, out_means + 1);
  *out_n = static_cast<int>(best.size());
  for (size_t i = 0; i < best.size(); ++i) {
    out_board[3 * i] = best[i].size;
    out_board[3 * i + 1] = best[i].a;
    out_board[3 * i + 2] = best[i].b;
  }
  return gap;
}

}  // extern "C"

namespace {

double solve_core(std::vector<Entry>& board, int q, int omega, int n_top,
                  int n_spare, double difficulty,
                  std::vector<Entry>* out_board, double* out_a, double* out_b,
                  double w_a, double w_b) {
  int n = static_cast<int>(board.size());
  int held_a = 0, held_b = 0, occupied_cliques = 0;
  for (int i = 0; i < n; ++i) {
    held_a += board[i].a;
    held_b += board[i].b;
    if (board[i].a + board[i].b > 0) ++occupied_cliques;
  }
  int sizes[2] = {omega, omega - 1};
  int supply[2] = {n_top, n_spare};
  std::vector<int> occ[2];
  for (int s = 0; s < 2; ++s)
    for (int i = 0; i < n; ++i)
      if (board[i].size == sizes[s]) occ[s].push_back(i);
  int freeCl[2];
  for (int s = 0; s < 2; ++s)
    freeCl[s] = std::max(0, supply[s] - static_cast<int>(occ[s].size()));

  int max_xmin = (held_a + held_b + q) / std::max(1, occupied_cliques) + 1;
  int q_total = held_b + q;
  double best = NEG;
  std::vector<Entry> best_board;
  double best_a = 0.0, best_b = 0.0;

  std::vector<int> order(n);
  for (int i = 0; i < n; ++i) order[i] = i;
  std::sort(order.begin(), order.end(), [&](int x, int y) {
    return board[x].a + board[x].b < board[y].a + board[y].b;
  });

  std::vector<int> add[2], fresh[2];
  for (int k_top = 0; k_top <= q; ++k_top) {
    int budget[2] = {k_top, q - k_top};
    for (int xmin = 1; xmin <= max_xmin; ++xmin) {
      // The anchor is the clique that ends at exactly xmin. B only adds, so a
      // clique already deeper than xmin can never be it -- those anchors are
      // provably redundant and enumerating them was most of the cost.
      std::vector<int> anchors;
      anchors.push_back(-1);
      anchors.push_back(-2);   // a newly opened clique attains xmin
      for (int i : order) {
        if (board[i].a + board[i].b <= xmin) anchors.push_back(i);
        else break;
      }
      for (int anchor : anchors) {
        bool ok = true;
        for (int s = 0; s < 2 && ok; ++s)
          ok = allocate(board, occ[s], freeCl[s], budget[s], xmin, q_total,
                        held_a, anchor, &add[s], &fresh[s], w_a, w_b);
        if (!ok) continue;
        std::vector<Entry> trial = board;
        for (int s = 0; s < 2; ++s) {
          for (size_t k = 0; k < occ[s].size(); ++k)
            trial[occ[s][k]].b += add[s][k];
          for (int depth : fresh[s])
            trial.push_back(Entry{sizes[s], 0, depth, 1});
        }
        double ma, mb;
        score_board(trial, difficulty, &ma, &mb);
        double obj = w_b * mb - w_a * ma;
        if (obj > best + 1e-12) {
          best = obj; best_board = trial; best_a = ma; best_b = mb;
        }
      }
    }
  }
  *out_board = best_board;
  *out_a = best_a;
  *out_b = best_b;
  return best;
}

std::vector<int> spread_even(int total, int slots) {
  std::vector<int> out;
  if (total <= 0 || slots <= 0) return out;
  int base = total / slots, extra = total % slots;
  for (int i = 0; i < slots; ++i) out.push_back(base + (i < extra ? 1 : 0));
  return out;
}

}  // namespace

// A's candidate boards. For a fixed omega/omega-1 split, A's own diversity term
// depends on the board only through the number of distinct cliques it occupies
// and the smallest count on any of them, so for each target minimum m the best
// board is the widest spread whose every count is at least m. Enumerating m
// therefore sweeps the whole (width, minimum) frontier; the old code emitted a
// single width per split, min(n_class, budget), which pins the minimum to 1.
std::vector<std::vector<Entry>> a_candidates(int q_a, int omega, int n_top,
                                             int n_spare) {
  std::vector<std::vector<Entry>> out;
  std::set<std::array<int, 3>> seen;
  for (int at_omega = 0; at_omega <= q_a; ++at_omega) {
    int rest = q_a - at_omega;
    if (at_omega > 0 && n_top <= 0) continue;
    if (rest > 0 && n_spare <= 0) continue;
    for (int m = 1; m <= q_a; ++m) {
      int j_top = at_omega > 0 ? std::max(1, std::min(n_top, at_omega / m)) : 0;
      int j_spare = rest > 0 ? std::max(1, std::min(n_spare, rest / m)) : 0;
      if (j_top == 0 && j_spare == 0) continue;
      std::array<int, 3> key{at_omega, j_top, j_spare};
      if (!seen.insert(key).second) continue;
      std::vector<Entry> board;
      for (int c : spread_even(at_omega, j_top))
        if (c > 0) board.push_back(Entry{omega, c, 0, 1});
      for (int c : spread_even(rest, j_spare))
        if (c > 0) board.push_back(Entry{omega - 1, c, 0, 1});
      if (!board.empty()) out.push_back(board);
    }
  }
  return out;
}

std::vector<std::array<int, 2>> board_key(const std::vector<Entry>& b) {
  std::vector<std::array<int, 2>> key;
  for (const auto& e : b)
    if (e.a > 0) key.push_back({e.size, e.a});
  std::sort(key.begin(), key.end());
  return key;
}

// Single-hotkey neighbours of a board: move one hotkey off some clique and onto
// another occupied clique, or onto a fresh one in either size class. The width
// sweep in a_candidates cannot reach a board whose counts are uneven inside a
// size class, and those are sometimes strictly better once B replies, so the
// family start is refined by steepest ascent over these moves.
std::vector<std::vector<Entry>> neighbours(const std::vector<Entry>& board,
                                           int omega, int n_top, int n_spare) {
  int used_top = 0, used_spare = 0;
  for (const auto& e : board) {
    if (e.a <= 0) continue;
    if (e.size == omega) ++used_top; else ++used_spare;
  }
  std::vector<Entry> dests = board;
  if (used_top < n_top) dests.push_back(Entry{omega, 0, 0, 1});
  if (used_spare < n_spare) dests.push_back(Entry{omega - 1, 0, 0, 1});

  std::vector<std::vector<Entry>> out;
  std::set<std::vector<std::array<int, 2>>> seen;
  seen.insert(board_key(board));
  for (size_t i = 0; i < board.size(); ++i) {
    if (board[i].a <= 0) continue;
    for (size_t j = 0; j < dests.size(); ++j) {
      if (j < board.size() && i == j) continue;
      std::vector<Entry> trial = board;
      // dests may hold board.size()+2 entries (a fresh omega AND a fresh
      // omega-1 slot), but trial grows by at most one, so the raw index j is
      // only valid for j < board.size(). Append and use the appended slot.
      size_t jj = j;
      if (j >= board.size()) { trial.push_back(dests[j]); jj = trial.size() - 1; }
      trial[i].a -= 1;
      trial[jj].a += 1;
      std::vector<Entry> kept;
      for (const auto& e : trial)
        if (e.a > 0) kept.push_back(e);
      if (kept.empty()) continue;
      if (!seen.insert(board_key(kept)).second) continue;
      out.push_back(kept);
    }
  }
  return out;
}

struct NoDeadline { bool operator()() const { return false; } };

template <class F, class G = NoDeadline>
std::vector<Entry> climb(std::vector<Entry> board, double* value, F eval,
                         int omega, int n_top, int n_spare, int max_iter,
                         G out_of_time = NoDeadline()) {
  for (int it = 0; it < max_iter; ++it) {
    if (out_of_time()) break;
    std::vector<std::vector<Entry>> cand = neighbours(board, omega, n_top, n_spare);
    const int n = static_cast<int>(cand.size());
    if (n == 0) break;
    std::vector<double> val(n, -1e18);
    volatile bool climb_stop = false;
#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < n; ++i) {
      if (climb_stop) continue;
      val[i] = eval(cand[i]);
      if (out_of_time()) climb_stop = true;
    }
    int arg = -1;
    double best_v = *value + 1e-12;
    for (int i = 0; i < n; ++i)
      if (val[i] > best_v) { best_v = val[i]; arg = i; }
    if (arg < 0) break;
    *value = val[arg];
    board = cand[arg];
  }
  return board;
}

void emit(const std::vector<Entry>& board, int* out_board, int* out_n) {
  *out_n = static_cast<int>(board.size());
  for (size_t i = 0; i < board.size(); ++i) {
    out_board[3 * i] = board[i].size;
    out_board[3 * i + 1] = board[i].a;
    out_board[3 * i + 2] = board[i].b;
  }
}

extern "C" {

double bs_bayes(int q_a, const int* ks, const double* ws, int n_k, int omega,
                int n_top, int n_spare, double difficulty, double fleet_a,
                double fleet_b, int* out_board, int* out_n) {
  auto eval = [&](const std::vector<Entry>& board) {
    double value = 0.0;
    for (int i = 0; i < n_k; ++i) {
      std::vector<Entry> work = board, reply;
      double ma = 0.0, mb = 0.0;
      // B optimises the POOLED margin, the same objective the sweep scores.
      // Leaving the weights at their defaults models an opponent maximising
      // mean_B - mean_A, which is a different player whenever the fleets differ.
      solve_core(work, ks[i], omega, n_top, n_spare, difficulty, &reply, &ma,
                 &mb, q_a / fleet_a, ks[i] / fleet_b);
      value += ws[i] * (q_a / fleet_a * ma - ks[i] / fleet_b * mb);
    }
    return value;
  };
  double best = -1e18;
  std::vector<Entry> best_board;
  for (const auto& board : a_candidates(q_a, omega, n_top, n_spare)) {
    double value = eval(board);
    if (value > best + 1e-12) { best = value; best_board = board; }
  }
  best_board = climb(best_board, &best, eval, omega, n_top, n_spare, 64);
  emit(best_board, out_board, out_n);
  return best;
}

double bs_maximin(int q_a, int q_b, int omega, int n_top, int n_spare,
                  double difficulty, double fleet_a, double fleet_b,
                  int* out_board, int* out_n, double deadline_s, int coarse,
                  int climb_iters) {
  const auto t0 = std::chrono::steady_clock::now();
  auto out_of_time = [&]() {
    if (deadline_s <= 0.0) return false;
    std::chrono::duration<double> d = std::chrono::steady_clock::now() - t0;
    return d.count() >= deadline_s;
  };
  auto eval = [&](const std::vector<Entry>& board) {
    std::vector<Entry> work = board, reply;
    double ma = 0.0, mb = 0.0;
    solve_core(work, q_b, omega, n_top, n_spare, difficulty, &reply, &ma, &mb,
               q_a / fleet_a, q_b / fleet_b);
    return q_a / fleet_a * ma - q_b / fleet_b * mb;
  };
  double best = -1e18;
  std::vector<Entry> best_board;
  std::vector<std::vector<Entry>> cands = a_candidates(q_a, omega, n_top, n_spare);
  const int nc = static_cast<int>(cands.size());
  std::vector<double> objs(nc, -1e18);
  std::vector<char> done(nc, 0);
  volatile bool stop = false;

  // COARSE-TO-FINE. a_candidates is a grid over (split, width) and the objective
  // is smooth along it, so scoring every cell is wasteful: at q_a=40 that is 410
  // full best-response evaluations, ~50ms each at low difficulty. Score every
  // stride-th cell, then rescore only the neighbourhood of the winner. The climb
  // that follows repairs anything the coarse pass stepped over.
  const int stride = (coarse > 1 && nc > 4 * coarse) ? coarse : 1;
  auto score_at = [&](int i) {
    if (done[i]) return;
    objs[i] = eval(cands[i]);
    done[i] = 1;
  };
  std::vector<int> pass;
  for (int i = 0; i < nc; i += stride) pass.push_back(i);
#pragma omp parallel for schedule(dynamic, 1)
  for (int p = 0; p < static_cast<int>(pass.size()); ++p) {
    if (stop) continue;
    score_at(pass[p]);
    if (out_of_time()) stop = true;
  }
  int arg = -1;
  for (int i = 0; i < nc; ++i)
    if (done[i] && objs[i] > best + 1e-12) { best = objs[i]; best_board = cands[i]; arg = i; }
  if (stride > 1 && arg >= 0 && !out_of_time()) {
    std::vector<int> fine;
    for (int i = std::max(0, arg - stride); i <= std::min(nc - 1, arg + stride); ++i)
      if (!done[i]) fine.push_back(i);
#pragma omp parallel for schedule(dynamic, 1)
    for (int p = 0; p < static_cast<int>(fine.size()); ++p) {
      if (stop) continue;
      score_at(fine[p]);
      if (out_of_time()) stop = true;
    }
    for (int i = 0; i < nc; ++i)
      if (done[i] && objs[i] > best + 1e-12) { best = objs[i]; best_board = cands[i]; }
  }
  // the candidate sweep alone is a valid answer; the climb only refines it, so
  // dropping it when the clock is gone costs quality, never correctness
  if (climb_iters > 0 && !out_of_time())
    best_board = climb(best_board, &best, eval, omega, n_top, n_spare, climb_iters,
                       out_of_time);
  emit(best_board, out_board, out_n);
  return best;
}

}  // extern "C"
