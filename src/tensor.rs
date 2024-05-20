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
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    pub fn get(&self, indices: &[usize]) -> f32 {
        let mut index = 0;
        for (i, &idx) in indices.iter().enumerate() {
            index += idx * self.strides[i];
        }
        self.data[index]
    }

    pub fn matmul(&self, other: &Tensor) -> Result<Tensor, String> {
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err("Matmul only supported for 2D tensors".into());
        }
        if self.shape[1] != other.shape[0] {
            return Err(format!("Dimension mismatch: {} != {}", self.shape[1], other.shape[0]));
        }

        let m = self.shape[0];