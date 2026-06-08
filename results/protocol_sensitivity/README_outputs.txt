Protocol-sensitivity output navigation
======================================

figures/performance/grouped_bars/                 grouped method bars, panels=protocols
figures/performance/protocol_lines/               one panel per method, lines=protocols
figures/performance/baseline_vs_tau/across_protocols/ baseline-only overlays by metric
figures/performance/baseline_vs_tau/by_protocol/  AP-style baseline degradation plots per protocol
figures/performance/method_advantage_focus/       robust-method delta vs own baseline at focus tau
figures/performance/combined_by_protocol/           Part-3-style combined body/all-metric plots per protocol
figures/mechanism/across_tau/                     final-epoch NTA/LNMR across noise rates
figures/mechanism/focus_tau/                      final-epoch focus-tau comparisons
figures/mechanism/epoch_trajectories/             cross-protocol NTA/LNMR over epochs
figures/mechanism/by_protocol/nta_lnmr/             Part-5-style combined NTA/LNMR plots per protocol
figures/mechanism/epoch_by_protocol/focus_tau/      Part-5-style two-panel epoch plot per protocol
figures/mechanism/epoch_by_protocol/grids/          all-tau epoch grids per protocol
figures/matrices/confusion/<protocol>/            raw and row-normalized confusion matrices
figures/matrices/confusion_delta/<protocol>/      focus-tau delta-vs-clean confusion matrices
figures/matrices/confusion_grid/<protocol>/       compact focus-tau row-normalized confusion grids
figures/matrices/perclass_f1/<protocol>/          per-class F1 heatmaps
figures/matrices/perclass_nta/<protocol>/         per-class NTA heatmaps
figures/matrices/perclass_lnmr/<protocol>/        per-class LNMR heatmaps
tables/performance/body/                          compact body tables
tables/performance/deltas/                        baseline-relative delta tables
tables/stats/                                     ranking, best-next, and interaction tables
tables/appendix/performance/                      full aggregate appendix tables
tables/mechanism/ and tables/appendix/mechanism/  mechanism tables
tables/matrices/<protocol>/                       focus-tau per-class appendix tables
data/                                             tidy CSVs behind outputs, grouped by purpose
