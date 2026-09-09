use super::diag::diversity;
use super::objective::{LinearMseObjective, Objective};

#[test]
fn linear_mse_valid_shape_preserves_fitness() {
    let obj = LinearMseObjective::new(vec![1.0, 2.0], vec![5.0], 1, 2, 1);
    assert_eq!(obj.fitness(&[1.0, 2.0]), 0.0);
}

#[test]
fn linear_mse_rejects_short_x_without_panicking() {
    let obj = LinearMseObjective::new(vec![1.0], vec![5.0], 1, 2, 1);
    assert_eq!(obj.fitness(&[1.0, 2.0]), f64::NEG_INFINITY);
}

#[test]
fn linear_mse_rejects_short_y_without_panicking() {
    let obj = LinearMseObjective::new(vec![1.0, 2.0], vec![], 1, 2, 1);
    assert_eq!(obj.fitness(&[1.0, 2.0]), f64::NEG_INFINITY);
}

#[test]
fn linear_mse_rejects_dimension_overflow_without_panicking() {
    let obj = LinearMseObjective::new(vec![], vec![], usize::MAX, 2, 2);
    assert_eq!(obj.fitness(&[0.0; 4]), f64::NEG_INFINITY);
}

#[test]
fn diversity_rejects_ragged_population_without_panicking() {
    let cands = vec![vec![1.0, 2.0], vec![3.0]];
    assert!(diversity(&cands).is_nan());
}

#[test]
fn diversity_rectangular_population_remains_finite() {
    let cands = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    assert!(diversity(&cands).is_finite());
}
