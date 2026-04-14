pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn new(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        Self { data: vec![0.0; size], shape }
    }

    pub fn matmul(&self, other: &Tensor) -> Result<Tensor, String> {
        // Matrix multiplication logic
        if self.shape[1] != other.shape[0] { return Err("Invalid dimensions".into()); }
        Ok(Tensor::new(vec![self.shape[0], other.shape[1]]))
    }
}