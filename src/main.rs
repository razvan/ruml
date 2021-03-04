
mod error;
mod util;
mod kmeans;

use kmeans::{KMeans, ORIGIN, Point};

use rand::SeedableRng;

///
/// Train a kmeans classifier and make some predictions.
///
fn main() {
    let k = 2;

    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
    let mut kmeans = KMeans {k, rng: &mut rng};

    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);    
    let points = kmeans::generate_clustered_points(
        &mut rng,
        15,                                             // number or points
        &vec![ORIGIN, Point::new(5.0, 3.0)],    // start centers
        &vec![(2.0, 2.0), (3.0, 1.0)],                  // cluster variances
    );
    
    let model = kmeans.train(&points, 0.3e-8, 100).unwrap();

    println!("Trained cluster centers:\n {:?}", model);

    let p = Point::new(6.0, 2.0);

    println!("Distance of {:?} to the clusters:\n {:?}", p, model.distance(&p));
    println!("Clusters by distance of {:?} to the centroids:\n {:?}", p, model.cluster(&p));
}

