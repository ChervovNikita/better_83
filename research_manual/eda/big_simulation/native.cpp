// Exact best responses for the two-player clique allocation game.
#include <algorithm>
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
  std::vector<Entry> live;
  int total = 0, omega = 0, x_min = 1 << 30;
  for (const Entry& e : board) {
    int held = e.a + e.b;
    if (e.mult <= 0 || held <= 0) continue;
    live.push_back(e);
    total += held * e.mult;
    omega = std::max(omega, e.size);
    x_min = std::min(x_min, held);
  }
  if (live.empty()) { *mean_a = *mean_b = 0.0; return; }
  std::vector<int> sizes;
  std::vector<double> opt;
  double scale = optimality_scale(live, total, omega, difficulty, sizes, opt);
  double sa = 0.0, sb = 0.0;
  long ca = 0, cb = 0;
  for (const Entry& e : live) {
    size_t k = std::find(sizes.begin(), sizes.end(), e.size) - sizes.begin();
    double r = opt[k] * scale + x_min / static_cast<double>(e.a + e.b);
    sa += e.a * e.mult * r; ca += e.a * e.mult;
    sb += e.b * e.mult * r; cb += e.b * e.mult;
  }
  *mean_a = ca ? sa / ca : 0.0;
  *mean_b = cb ? sb / cb : 0.0;
}

struct Item { int kind, index, low, high; };   // kind 0 occupied, 1 fresh

double gain_of(const std::vector<Entry>& board, const Item& it, int take,
               int xmin, int q, int held_a) {
  if (it.kind == 1) return take ? xmin / static_cast<double>(q) : 0.0;
  const Entry& e = board[it.index];
  int total = e.a + e.b + take;
  return xmin * take / static_cast<double>(total) / static_cast<double>(q)
       - xmin * e.a / static_cast<double>(total) / static_cast<double>(held_a);
}

bool allocate(const std::vector<Entry>& board, const std::vector<int>& occupied,
              int freeCliques, int budget, int xmin, int q, int held_a,
              int anchor, std::vector<int>* add, std::vector<int>* fresh) {
  std::vector<Item> items;
  int lo_total = 0;
  for (int i : occupied) {
    int low = std::max(0, xmin - board[i].a - board[i].b);
    int high = (i == anchor) ? low : budget;
    items.push_back(Item{0, i, low, high});
    lo_total += low;
  }
  int n_fresh = xmin > 0 ? std::min(freeCliques, budget / xmin) : 0;
  for (int j = 0; j < n_fresh; ++j) items.push_back(Item{1, -1, xmin, budget});
  add->assign(occupied.size(), 0);
  fresh->clear();
  if (items.empty()) return budget == 0;
  if (lo_total > budget) return false;

  int n = static_cast<int>(items.size());
  std::vector<double> table((n + 1) * (budget + 1), NEG);
  std::vector<int> back((n + 1) * (budget + 1), 0);
  table[0] = 0.0;
  for (int step = 0; step < n; ++step) {
    const Item& it = items[step];
    for (int spent = 0; spent <= budget; ++spent) {
      double best = NEG;
      int best_take = 0;
      int hi = std::min(spent, it.high);
      for (int take = 0; take <= hi; ++take) {
        if (take < it.low && !(it.kind == 1 && take == 0)) continue;
        if (it.kind == 1 && take > 0 && take < xmin) continue;
        double prev = table[step * (budget + 1) + (spent - take)];
        if (prev <= NEG / 2) continue;
        double v = prev + gain_of(board, it, take, xmin, q, held_a);
        if (v > best) { best = v; best_take = take; }
      }
      table[(step + 1) * (budget + 1) + spent] = best;
      back[(step + 1) * (budget + 1) + spent] = best_take;
    }
  }
  if (table[n * (budget + 1) + budget] <= NEG / 2) return false;
  std::vector<int> takes(n, 0);
  int spent = budget;
  for (int step = n; step > 0; --step) {
    takes[step - 1] = back[step * (budget + 1) + spent];
    spent -= takes[step - 1];
  }
  for (size_t k = 0; k < occupied.size(); ++k) (*add)[k] = takes[k];
  for (size_t k = occupied.size(); k < takes.size(); ++k)
    if (takes[k] > 0) fresh->push_back(takes[k]);
  return true;
}

}  // namespace

extern "C" {

void bs_score(const int* flat, int n, double difficulty, double* out) {
  std::vector<Entry> board(n);
  for (int i = 0; i < n; ++i)
    board[i] = Entry{flat[4 * i], flat[4 * i + 1], flat[4 * i + 2],
                     flat[4 * i + 3]};
  score_board(board, difficulty, out, out + 1);
}

double bs_best_response(const int* flat, int n, int q, int omega, int n_top,
                        int n_spare, double difficulty, int* out_board,
                        int* out_n, double* out_means) {
  std::vector<Entry> board(n);
  int held_a = 0, held_b = 0, occupied_cliques = 0;
  for (int i = 0; i < n; ++i) {
    board[i] = Entry{flat[3 * i], flat[3 * i + 1], flat[3 * i + 2], 1};
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

  std::vector<int> anchors;
  anchors.push_back(-1);
  {
    std::vector<int> order(n);
    for (int i = 0; i < n; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int x, int y) {
      return board[x].a + board[x].b < board[y].a + board[y].b;
    });
    for (int i : order) anchors.push_back(i);
  }

  std::vector<int> add[2], fresh[2];
  for (int k_top = 0; k_top <= q; ++k_top) {
    int budget[2] = {k_top, q - k_top};
    for (int xmin = 1; xmin <= max_xmin; ++xmin) {
      for (int anchor : anchors) {
        bool ok = true;
        for (int s = 0; s < 2 && ok; ++s)
          ok = allocate(board, occ[s], freeCl[s], budget[s], xmin, q_total,
                        held_a, anchor, &add[s], &fresh[s]);
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
        if (mb - ma > best + 1e-12) {
          best = mb - ma; best_board = trial; best_a = ma; best_b = mb;
        }
      }
    }
  }
  *out_n = static_cast<int>(best_board.size());
  for (size_t i = 0; i < best_board.size(); ++i) {
    out_board[3 * i] = best_board[i].size;
    out_board[3 * i + 1] = best_board[i].a;
    out_board[3 * i + 2] = best_board[i].b;
  }
  out_means[0] = best_a;
  out_means[1] = best_b;
  return best;
}

}  // extern "C"
