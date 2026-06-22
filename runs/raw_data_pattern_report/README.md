# Raw Data Pattern Report

Data: `src/npz_e65_data/E65_data.npz`

## Key outputs
- `activity_and_pca.png`
- `position_tuning.png`
- `neuron_correlations.png`
- `pca_behavior_structure.png`
- `population_distance_relationships.png`

## Numeric summary
```json
{
  "shape": {
    "raw_frames": 7434,
    "raw_neurons": 375,
    "active_neurons": 357,
    "globally_silent_neurons": 18,
    "active_frames_mind": 7415,
    "silent_frames": 19
  },
  "sparsity": {
    "zero_fraction_all_active_neurons": 0.9841021154224402,
    "positive_fraction_all_active_neurons": 0.015897884577559838,
    "median_neuron_positive_frame_fraction": 0.005515200430454668,
    "mean_neuron_positive_frame_fraction": 0.015897884577559838,
    "median_frame_active_neuron_count": 6.0,
    "mean_frame_active_neuron_count": 5.675544794188862,
    "max_frame_active_neuron_count": 18,
    "median_positive_value": 0.41531848907470703,
    "p95_positive_value": 1.4825372695922852,
    "p99_positive_value": 2.0787413120269775
  },
  "trials": {
    "n_trials": 188,
    "frames_per_trial_min_median_max": [
      29,
      38.0,
      78
    ],
    "duration_s_min_median_max": [
      1.9411580264568329,
      2.564686819911003,
      5.347037374973297
    ]
  },
  "pca": {
    "pcs_for_0.50": 21,
    "pcs_for_0.75": 51,
    "pcs_for_0.80": 61,
    "pcs_for_0.90": 94,
    "pcs_for_0.95": 129,
    "pcs_for_0.99": 151,
    "variance_2_pcs": 0.10593602066384214,
    "variance_3_pcs": 0.14335879928390313,
    "variance_5_pcs": 0.20895804396857495,
    "variance_7_pcs": 0.2650183757438294,
    "variance_10_pcs": 0.3322097731531375,
    "variance_20_pcs": 0.48878235151488614,
    "variance_50_pcs": 0.747577035995537,
    "variance_80_pcs": 0.8683015949469888,
    "variance_100_pcs": 0.9127057003391413,
    "variance_125_pcs": 0.9462707997188421,
    "variance_150_pcs": 0.9673985291188667
  },
  "neuron_correlations": {
    "median_corr": -0.0032784278353633,
    "mean_corr": 0.001127699918918988,
    "p95_abs_corr": 0.0370029728178644,
    "p99_abs_corr": 0.1495599661159821,
    "frac_abs_corr_gt_0p25": 0.00413873414534353,
    "frac_abs_corr_gt_0p5": 0.0009441979038806534,
    "frac_positive_corr": 0.10375161300475247
  },
  "position_tuning": {
    "position_range": [
      -4.539243221282959,
      301.4441223144531
    ],
    "bin_counts_min_median_max": [
      173,
      196.0,
      824
    ],
    "median_eta2": 0.006993770599365234,
    "mean_eta2": 0.013650625944137573,
    "frac_eta2_gt_0p05": 0.04201680672268908,
    "frac_eta2_gt_0p10": 0.0056022408963585435,
    "median_split_half_corr": 0.3857896846418503,
    "frac_split_half_corr_gt_0p3": 0.453781512605042,
    "frac_split_half_corr_gt_0p5": 0.36694677871148457
  },
  "population_geometry": {
    "corr_pca10_distance_with_position_distance": 0.004410700003233023,
    "corr_pca10_distance_with_time_distance": 0.043637841551981055,
    "mean_distance_same_trial": 0.9008923217655528,
    "mean_distance_diff_trial": 1.1488092549159399,
    "mean_distance_same_maze": 1.1188231896842062,
    "mean_distance_diff_maze": 1.297183463469539,
    "mean_distance_same_choice": 1.1058990730724796,
    "mean_distance_diff_choice": 1.189607231949568
  },
  "temporal_structure": {
    "median_population_corr_by_lag_frames": {
      "1": 0.9758904576301575,
      "2": 0.9190846681594849,
      "3": 0.8400326371192932,
      "5": 0.6337893605232239,
      "10": 0.20097050070762634
    },
    "median_population_l2_delta_by_lag_frames": {
      "1": 0.37695908546447754,
      "2": 0.6674448251724243,
      "3": 0.9236559271812439,
      "5": 1.3435486555099487,
      "10": 1.878729224205017
    }
  },
  "behavior_decoding_from_pca20": {
    "choice_balanced_accuracy_from_pca20": 0.6886695494648452,
    "correct_balanced_accuracy_from_pca20": 0.598265961160698,
    "trial_type_balanced_accuracy_from_pca20": 0.6809758577201588,
    "maze_id_balanced_accuracy_from_pca20": 0.6861557214459406,
    "position_r2_from_pca20": 0.47636713564817523,
    "velocity_r2_from_pca20": 0.3956183602404416,
    "evidence_r2_from_pca20": 0.2361581643085265
  },
  "top_position_tuned_neurons": [
    1,
    22,
    17,
    0,
    3,
    5,
    127,
    6,
    2,
    54,
    34,
    37,
    9,
    35,
    56,
    100,
    128,
    11,
    20,
    52
  ],
  "top_reliable_tuning_neurons": [
    331,
    226,
    287,
    1,
    100,
    58,
    127,
    37,
    56,
    13,
    244,
    11,
    47,
    50,
    60,
    17,
    119,
    22,
    9,
    128
  ],
  "top_correlated_pairs_original_neuron_ids": [
    {
      "neuron_a": 308,
      "neuron_b": 318,
      "corr": 0.9513283636524372
    },
    {
      "neuron_a": 234,
      "neuron_b": 314,
      "corr": 0.9223448567060797
    },
    {
      "neuron_a": 178,
      "neuron_b": 187,
      "corr": 0.8908107323671516
    },
    {
      "neuron_a": 72,
      "neuron_b": 94,
      "corr": 0.8805965913885012
    },
    {
      "neuron_a": 294,
      "neuron_b": 352,
      "corr": 0.8371222107945282
    },
    {
      "neuron_a": 213,
      "neuron_b": 294,
      "corr": 0.8355669454922714
    },
    {
      "neuron_a": 81,
      "neuron_b": 153,
      "corr": 0.822001719778749
    },
    {
      "neuron_a": 309,
      "neuron_b": 323,
      "corr": 0.8207273445869951
    },
    {
      "neuron_a": 274,
      "neuron_b": 309,
      "corr": 0.8056162647494928
    },
    {
      "neuron_a": 213,
      "neuron_b": 352,
      "corr": 0.8027033437775054
    },
    {
      "neuron_a": 54,
      "neuron_b": 134,
      "corr": 0.7951648948736642
    },
    {
      "neuron_a": 82,
      "neuron_b": 146,
      "corr": 0.7919282752325251
    },
    {
      "neuron_a": 92,
      "neuron_b": 211,
      "corr": 0.790185676054982
    },
    {
      "neuron_a": 117,
      "neuron_b": 273,
      "corr": 0.784085794840316
    },
    {
      "neuron_a": 104,
      "neuron_b": 250,
      "corr": 0.7302290680880549
    },
    {
      "neuron_a": 259,
      "neuron_b": 291,
      "corr": 0.7269531285015637
    },
    {
      "neuron_a": 274,
      "neuron_b": 323,
      "corr": 0.723886363194458
    },
    {
      "neuron_a": 272,
      "neuron_b": 342,
      "corr": 0.7152526632113381
    },
    {
      "neuron_a": 129,
      "neuron_b": 143,
      "corr": 0.6901804015691277
    },
    {
      "neuron_a": 69,
      "neuron_b": 191,
      "corr": 0.6897528185503579
    }
  ]
}
```