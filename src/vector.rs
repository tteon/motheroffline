use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Vector {
    pub id: usize,
    #[serde(rename = "data")]
    pub data: Vec<f32>,
}

impl Vector {
    pub fn new(id: usize, data: Vec<f32>) -> Self {
        Self { id, data }
    }
}
