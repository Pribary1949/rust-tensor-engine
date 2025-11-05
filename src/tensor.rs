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
        let k = self.shape[1];
        let n = other.shape[1];
        let mut result_data = vec![0.0; m * n];

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for p in 0..k {
                    sum += self.data[i * k + p] * other.data[p * n + j];
                }
                result_data[i * n + j] = sum;
            }
        }

        Ok(Tensor::new(result_data, vec![m, n]))
    }

    pub fn relu(&self) -> Tensor {
        let data = self.data.iter().map(|&x| if x > 0.0 { x } else { 0.0 }).collect();
        Tensor::new(data, self.shape.clone())
    }
}

impl Add for Tensor {
    type Output = Result<Tensor, String>;

    fn add(self, other: Self) -> Self::Output {
        if self.shape != other.shape {
            return Err("Shape mismatch for addition".into());
        }
        let data = self.data.iter().zip(other.data.iter()).map(|(a, b)| a + b).collect();
        Ok(Tensor::new(data, self.shape))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.data, vec![19.0, 22.0, 43.0, 50.0]);
    }
}
