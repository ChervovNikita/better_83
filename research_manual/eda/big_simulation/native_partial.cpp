// Expected-value best response when only the occupancy multiset is known.
#include <algorithm>
#include <cmath>
#include <map>
#include <vector>

namespace {

struct Entry { int size, a, b, mult; };

void score_board(const std::vector<Entry>& board, double difficulty,
                 double* mean_a, double* mean_b) {
  int total = 0, omega = 0, x_min = 1 << 30;
  std::vector<const Entry*> live;
  for (const Entry& e : board) {
    int held = e.a + e.b;
    if (e.mult <= 0 || held <= 0) continue;
    live.push_back(&e);
    total += held * e.mult;
    omega = std::max(omega, e.size);
    x_min = std::min(x_min, held);
  }
  if (live.empty()) { *mean_a = *mean_b = 0.0; return; }
  std::map<int, int, std::greater<int> > bigger;
  for (const Entry* e : live) bigger[e->size] += (e->a + e->b) * e->mult;
  std::map<int, double> opt;
  double inv = 1.0 / static_cast<double>(total), top = 0.0;
  int run = 0;
  for (std::map<int, int, std::greater<int> >::iterator it = bigger.begin();
       it != bigger.end(); ++it) {
    double rel = static_cast<double>(it->first) / static_cast<double>(omega);
    double v = std::exp(-(run * inv) / rel);
    opt[it->first] = v;
    if (v > top) top = v;
    run += it->second;
  }
  double scale = (1.0 + difficulty) / top;
  double sa = 0.0, sb = 0.0;
  long ca = 0, cb = 0;
  for (const Entry* e : live) {
    double r = opt[e->size] * scale + x_min / static_cast<double>(e->a + e->b);
    sa += e->a * e->mult * r; ca += e->a * e->mult;
    sb += e->b * e->mult * r; cb += e->b * e->mult;
  }
  *mean_a = ca ? sa / ca : 0.0;
  *mean_b = cb ? sb / cb : 0.0;
}

struct Table { std::vector<std::vector<int> > cell; double prob; };

void compositions(int total, int parts, std::vector<int>& cur,
                  std::vector<std::vector<int> >* out) {
  if (parts == 1) { cur.push_back(total); out->push_back(cur); cur.pop_back(); return; }
  for (int head = 0; head <= total; ++head) {
    cur.push_back(head);
    compositions(total - head, parts - 1, cur, out);
    cur.pop_back();
  }
}

void walk(const std::vector<int>& rows, const std::vector<int>& cols, int i,
          std::vector<int> remaining, std::vector<std::vector<int> > acc,
          double log_num, double log_den, std::vector<Table>* out) {
  if (i == static_cast<int>(rows.size()) - 1) {
    int s = 0;
    for (int c : remaining) { if (c < 0) return; s += c; }
    if (s != rows[i]) return;
    acc.push_back(remaining);
    double w = log_num - log_den;
    for (const std::vector<int>& row : acc)
      for (int x : row) w -= std::lgamma(x + 1.0);
    Table t; t.cell = acc; t.prob = std::exp(w);
    out->push_back(t);
    return;
  }
  std::vector<std::vector<int> > splits;
  std::vector<int> cur;
  compositions(rows[i], static_cast<int>(cols.size()), cur, &splits);
  for (const std::vector<int>& sp : splits) {
    bool ok = true;
    for (size_t j = 0; j < sp.size(); ++j)
      if (sp[j] > remaining[j]) { ok = false; break; }
    if (!ok) continue;
    std::vector<int> rem = remaining;
    for (size_t j = 0; j < sp.size(); ++j) rem[j] -= sp[j];
    std::vector<std::vector<int> > acc2 = acc;
    acc2.push_back(sp);
    walk(rows, cols, i + 1, rem, acc2, log_num, log_den, out);
  }
}

void tables(const std::vector<int>& rows, const std::vector<int>& cols,
            std::vector<Table>* out) {
  int n = 0;
  for (int r : rows) n += r;
  double log_den = std::lgamma(n + 1.0), log_num = 0.0;
  for (int r : rows) log_num += std::lgamma(r + 1.0);
  for (int c : cols) log_num += std::lgamma(c + 1.0);
  out->clear();
  walk(rows, cols, 0, cols, std::vector<std::vector<int> >(), log_num, log_den,
       out);
}

struct Klass {
  std::vector<int> values_a, mult_a, values_b, mult_b;
  std::vector<Table> tab;
  int size;
};

void classify(const std::vector<int>& counts, const std::vector<int>& assign,
              Klass* k) {
  std::map<int, int> la, lb;
  for (int c : counts) la[c]++;
  for (int m : assign) lb[m]++;
  k->values_a.clear(); k->mult_a.clear();
  for (std::map<int, int>::iterator it = la.begin(); it != la.end(); ++it) {
    k->values_a.push_back(it->first); k->mult_a.push_back(it->second);
  }
  k->values_b.clear(); k->mult_b.clear();
  for (std::map<int, int>::iterator it = lb.begin(); it != lb.end(); ++it) {
    k->values_b.push_back(it->first); k->mult_b.push_back(it->second);
  }
  tables(k->mult_a, k->mult_b, &k->tab);
}

}  // namespace

extern "C" {

void bsp_expected(const int* counts_top, const int* assign_top, int n_top,
                  const int* counts_sp, const int* assign_sp, int n_sp,
                  const int* fresh_size, const int* fresh_depth, int n_fresh,
                  int omega, double difficulty, double* out) {
  std::vector<Klass> classes;
  if (n_top > 0) {
    Klass k;
    k.size = omega;
    classify(std::vector<int>(counts_top, counts_top + n_top),
             std::vector<int>(assign_top, assign_top + n_top), &k);
    classes.push_back(k);
  }
  if (n_sp > 0) {
    Klass k;
    k.size = omega - 1;
    classify(std::vector<int>(counts_sp, counts_sp + n_sp),
             std::vector<int>(assign_sp, assign_sp + n_sp), &k);
    classes.push_back(k);
  }
  std::vector<Entry> fresh;
  for (int i = 0; i < n_fresh; ++i)
    fresh.push_back(Entry{fresh_size[i], 0, fresh_depth[i], 1});

  std::vector<size_t> idx(classes.size(), 0);
  double ea = 0.0, eb = 0.0;
  bool done = classes.empty();
  if (classes.empty()) {
    double ma, mb;
    score_board(fresh, difficulty, &ma, &mb);
    out[0] = ma; out[1] = mb;
    return;
  }
  while (!done) {
    double w = 1.0;
    std::vector<Entry> board = fresh;
    for (size_t c = 0; c < classes.size(); ++c) {
      const Klass& k = classes[c];
      const Table& t = k.tab[idx[c]];
      w *= t.prob;
      for (size_t i = 0; i < t.cell.size(); ++i)
        for (size_t j = 0; j < t.cell[i].size(); ++j)
          if (t.cell[i][j])
            board.push_back(Entry{k.size, k.values_a[i], k.values_b[j],
                                  t.cell[i][j]});
    }
    double ma, mb;
    score_board(board, difficulty, &ma, &mb);
    ea += w * ma; eb += w * mb;
    size_t c = 0;
    while (true) {
      if (++idx[c] < classes[c].tab.size()) break;
      idx[c] = 0;
      if (++c == classes.size()) { done = true; break; }
    }
  }
  out[0] = ea; out[1] = eb;
}

}  // extern "C"
