use std::ops::{Add, Mul};

#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let strides = Self::compute_strides(&shape);
        assert_eq!(data.len(), shape.iter().product::<usize>(), "Data length mismatch");
        Tensor { data, shape, strides }
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        Self::new(vec![0.0; size], shape)
    }

    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];