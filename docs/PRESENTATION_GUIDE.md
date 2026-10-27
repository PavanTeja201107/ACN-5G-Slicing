# End-semester presentation guide

1. **Problem recap (1 slide).** Static 50/30/20 wastes capacity; reactive control lags bursts; reconfiguration has a cost.
2. **ML formulation.** Predict each slice's demand at t+1 (Mbps) from demand history. The RF predicts demand, **not**
   allocation - the original controller (fairness, hysteresis, cooldown, cost) still decides.
3. **Features.** Slice one-hot, 10 lags, MA5/MA10/EWMA, std, 1- and 5-step differences, range, and all three
   slices' current demand. Say why allocation/QoS features were excluded (demand is exogenous in the simulator).
4. **Why Random Forest.** CPU-only, small (7 MB), fast to train (about 1 min), non-linear (handles bursts), gives feature
   importances, easy to explain; no GPU/LSTM needed.
5. **Data and split.** Synthetic: original six scenarios perturbed per seed; train/val/test by seed; no leakage
   (explain t -> t+1, end-of-trace rows dropped). Be explicit that it is modelled, not real 5G.
6. **Baselines.** A static, B moving average (the proposal's "EWMA/MA"), C RF in the same controller; same stream,
   seeds, thresholds, cooldown.
7. **Prediction results.** RF MAE 0.88 vs MA 2.44 Mbps on held-out seeds (0.62 vs 2.11 on the original stream).
   Show `plots/ml/actual_vs_predicted_canonical_stream.png` (MA lags bursts).
8. **Network results.** Original stream: violations 368 / 301 / 298. 10 streams: 335.7 / 290.3 / 272.8; RF fewer in 10/10.
   Say plainly: gain is **small**; static wins on some seeds; URLLC p95 and fairness are worse than static.
9. **Why accuracy did not translate into big QoS gains.** Controller input differs by about 1 Mbps; step 5 and cooldown 10
   dominate; identical reconfiguration timing. Suggests tuning the controller (smaller steps, shorter cooldown) as future work.
10. **Honest limitations.** Synthetic data from shared templates; leave-one-scenario-out shows no consistent edge over
    persistence on unseen scenarios; simulation only, no Open5GS/UERANSIM validation; no significance test.
11. **Likely questions.** *Is the MA really EWMA?* No - 5-sample simple moving average (state it first). *Is the test leaked?*
    Seeds disjoint, but templates shared (hence leave-scenario-out). *Why not LSTM?* Out of scope, CPU/lightweight goal.
    *Does RF beat MA?* In prediction yes; in QoS only modestly.
