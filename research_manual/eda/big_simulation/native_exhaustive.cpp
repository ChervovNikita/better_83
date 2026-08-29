// Exhaustive search over EVERY board A can commit, for rounds small enough to
// afford it. Included rather than linked so it shares one definition of the
// scorer and the responder with native.cpp.
#include "native.cpp"

namespace {

// Partitions of `total` into at most `slots` positive non-increasing parts.
void partitions(int total, int slots, int cap,
                std::vector<std::vector<int>>* out, std::vector<int>* acc) {
  if (total == 0) { out->push_back(*acc); return; }
  if (slots == 0) return;
  for (int head = std::min(cap, total); head >= 1; --head) {
    acc->push_back(head);
    partitions(total - head, slots - 1, head, out, acc);
    acc->pop_back();
  }
}

std::vector<std::vector<int>> all_partitions(int total, int slots) {
  std::vector<std::vector<int>> out;
  std::vector<int> acc;
  if (total == 0) { out.push_back(acc); return out; }
  partitions(total, slots, total, &out, &acc);
  return out;
}

}  // namespace

extern "C" {

// Returns the best objective over every A board, or NaN if the enumeration
// would exceed `budget` boards. Writes the winning board.
double bs_exhaustive_a(int q_a, int q_b, int omega, int n_top, int n_spare,
                       double difficulty, double fleet_a, double fleet_b,
                       long budget, int* out_board, int* out_n,
                       long* out_count) {
  std::vector<std::vector<std::vector<int>>> tops(q_a + 1), spares(q_a + 1);
  long count = 0;
  for (int a = 0; a <= q_a; ++a) {
    tops[a] = all_partitions(a, std::min(n_top, a));
    spares[q_a - a] = all_partitions(q_a - a, std::min(n_spare, q_a - a));
    count += static_cast<long>(tops[a].size()) * spares[q_a - a].size();
    if (count > budget) { *out_count = count; return std::nan(""); }
  }
  *out_count = count;

  double w_a = q_a / fleet_a, w_b = q_b / fleet_b;
  double best = -1e18;
  std::vector<Entry> best_board;
  for (int a = 0; a <= q_a; ++a) {
    for (const auto& top : tops[a]) {
      for (const auto& spare : spares[q_a - a]) {
        std::vector<Entry> board;
        for (int c : top) board.push_back(Entry{omega, c, 0, 1});
        for (int c : spare) board.push_back(Entry{omega - 1, c, 0, 1});
        if (board.empty()) continue;
        std::vector<Entry> work = board, reply;
        double ma = 0.0, mb = 0.0;
        solve_core(work, q_b, omega, n_top, n_spare, difficulty, &reply, &ma,
                   &mb);
        double obj = w_a * ma - w_b * mb;
        if (obj > best + 1e-12) { best = obj; best_board = board; }
      }
    }
  }
  emit(best_board, out_board, out_n);
  return best;
}

}  // extern "C"
