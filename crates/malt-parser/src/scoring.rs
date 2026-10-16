//! MCSVM_CS prediction: `MaltLiblinearModel.predict_one` — f64 accumulation
//! over the feature vector in its built order, first-strictly-greater argmax
//! (exact ties keep the earlier class, matching Java's `>`).

use crate::model::{DecisionModel, FeatureNode};

/// Class labels in predictor preference order (argmax first; the full
/// ranking feeds the legality fallback).
pub fn predict(model: &DecisionModel, x: &[FeatureNode]) -> Vec<i32> {
    let nr_class = model.labels.len();
    let mut dec_values = vec![0.0f64; nr_class];
    let n = if model.bias >= 0.0 {
        model.nr_feature + 1
    } else {
        model.nr_feature
    } as i32;

    for node in x {
        if node.index <= n {
            // Our `w` is 1-based (row 0 unused); Java's is 0-based with
            // `t = index - 1` — same row.
            if let Some(row) = model.w.get(node.index as usize) {
                for &(col, weight) in row {
                    if (col as usize) < nr_class {
                        dec_values[col as usize] += weight * node.value;
                    }
                }
            }
        }
    }

    let mut prediction: Vec<usize> = (0..nr_class).collect();
    // Java sorts via selection sort with strict `>`: ties keep label order.
    for i in 0..nr_class.saturating_sub(1) {
        let mut i_max = i;
        for j in (i + 1)..nr_class {
            if dec_values[j] > dec_values[i_max] {
                i_max = j;
            }
        }
        if i_max != i {
            dec_values.swap(i, i_max);
            prediction.swap(i, i_max);
        }
    }
    prediction.into_iter().map(|i| model.labels[i]).collect()
}
