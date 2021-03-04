use thiserror::Error;

#[derive(Debug, Error)]
pub enum KMeansError {
    #[error("K is too small or too large for the given data set.")]
    InvalidK
}

pub type KMeansResult<T> = std::result::Result<T, KMeansError>;