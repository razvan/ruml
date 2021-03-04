

use crate::error;
use crate::util;

use rand::Rng;

///
/// Utility function to generate clustered samples.
///
pub fn generate_clustered_points(
    rng: & mut impl Rng,
    amount: usize,
    centroids: &Vec<Point>,
    variance: &Vec<(f32, f32)>,
) -> Vec<Point> {
    let mut result = Vec::with_capacity(amount as usize);
    let mut lop = 0;

    if centroids.len() != variance.len() {
        panic!("Centroid and variance arrays are not the same length!");
    }

    for _ in 0..amount {
        let p: &Point = &centroids[lop];

        result.push(Point {
            x1: rng.gen_range(-1.0..1.0) * variance[lop].0 + p.x1,
            x2: rng.gen_range(-1.0..1.0) * variance[lop].1 + p.x2,
        });

        lop = (lop + 1) % centroids.len();
    }

    result
}

///
/// A two dimensional instance.
/// TODO: make it n-dimensional
///
#[derive(Debug, Copy, Clone)]
pub struct Point {
    x1: f32,
    x2: f32,
}

pub const ORIGIN: Point = Point { x1: 0.0, x2: 0.0 };

impl Point {
    pub fn new(x1: f32, x2: f32) -> Point {
        Point {x1, x2}
    }

    fn add(self, other: &Point) -> Point {
        Point {
            x1: self.x1 + other.x1,
            x2: self.x2 + other.x2,
        }
    }

    fn sub(self, other: &Point) -> Point {
        Point {
            x1: self.x1 - other.x1,
            x2: self.x2 - other.x2,
        }
    }

    fn div(self, divisor: f32) -> Point {
        Point {
            x1: self.x1 / divisor,
            x2: self.x2 / divisor,
        }
    }

    fn powi(self, i: i32) -> Point {
        Point {
            x1: self.x1.powi(i),
            x2: self.x2.powi(i),
        }
    }

    fn distance(a: &Point, b: &Point) -> f32 {
        ((a.x1 - b.x1).powi(2) + (a.x2 - b.x2).powi(2)).sqrt()
    }

    ///
    /// Euclidian norm of the distance between a and b.
    ///
    fn distance_norm(a: &[Point], b: &[Point]) -> f32 {
        a.iter().zip(b.iter())
            .map(|(i, j)| i.sub(&j).powi(2))
            .fold(0.0, |sum, p| sum + p.x1 + p.x2)
            .sqrt()
    }

    fn centroid(points: &[Point]) -> Point {
        let pl = points.len() as f32;
        if points.is_empty() {
            ORIGIN
        } else {
            points.iter().fold(ORIGIN, |s, p| s.add(p)).div(pl)
        }
    }
}

pub struct KMeans<'a, R: Rng> {
    pub k: u32,
    pub rng: &'a mut R,
}

impl<'a, R: Rng> KMeans<'a, R> {

    pub fn train(&mut self, points: &[Point], tol: f32, max_iter: u32) -> error::KMeansResult<KMeansModel> {
        if self.k > points.len() as u32 {
            Err(error::KMeansError::InvalidK)
        } else {
            // 0. init clusters
            let mut centroids = self.init_centroids(points);
            let mut iter: u32 = 0;
            let mut done = false;

            // 1. while tol || max_iter
            while iter < max_iter && !done {

                // 2. assign points to clusters
                let clusters = Self::assign_points_to_clusters(&centroids, points);

                // 3. recompute centroids, euclidian norm between old and new center distances (tol)
                let new_centroids: Vec<Point> = clusters.into_iter().map(|c| Point::centroid(&c)).collect();

                // advance
                let center_distance_norm = Point::distance_norm(&centroids, &new_centroids);
                if tol > center_distance_norm {
                    println!("Stop training because cluster change is too small: {}", center_distance_norm);
                    done = true;
                }
                iter += 1;

                centroids = new_centroids;
            }
            println!("Stopped after {} iterations", iter);
            Ok(KMeansModel {centroids: centroids})
        }
    }

    ///
    /// Returns a vec of clusters with the same length as the given centroids.
    /// Each item (cluster) is again a Vec of Points
    /// assigned to the centroid at the same index in the input.
    /// Some items (clusters) may be empty if no Points were assigned to that cluster.
    ///
    fn assign_points_to_clusters(centroids: &[Point], points: &[Point]) -> Vec<Vec<Point>> {
        // TODO: better handling of empty centroids and/or points

        // Initialize result
        let mut clusters: Vec<Vec<Point>> = Vec::with_capacity(centroids.len());
        for _ in 0..centroids.len() {
            clusters.push(Vec::new());
        }

        // Map points to clusters
        for p in points {
            // index of the nearest cluster and the distance to it's center
            let (minindex, _) = centroids
                .iter()
                .map(|&c| Point::distance(&p, &c))
                .enumerate()
                .min_by(|&(_i, vi), &(_j, vj)| util::cmp_f32(vi, vj))
                .unwrap();
            clusters[minindex].push(points[minindex]);
        }
        clusters
    }

    fn init_centroids(&mut self, points: &[Point]) -> Vec<Point> {
        let mut result: Vec<Point> = Vec::with_capacity(self.k as usize);
        for _ in 0..self.k {
            // TODO: ensure no duplicates are selected
            result.push( points[ (self.rng.next_u32() as usize) % points.len()]);
        }
        result
    }

}

#[derive(Debug)]
pub struct KMeansModel {
    centroids: Vec<Point>,
}

impl KMeansModel {
    ///
    /// Compute the distance of p to all clusters.
    ///
    pub fn distance(&self, p: &Point) -> Vec<f32> {
        self.centroids
            .iter()
            .map(|&c| Point::distance(p, &c))
            .collect()
    }

    ///
    /// Utility function that returns the cluster indexes sorted by the
    /// distance from p to their centers.
    ///
    pub fn cluster(&self, p: &Point) -> Vec<usize> {
        let mut idist: Vec<(usize, f32)> = self.distance(p)
            .into_iter()
            .enumerate()
            .collect();

        idist.sort_by(|(_, di), (_, dj)| di.partial_cmp(dj).unwrap());

        idist.into_iter().map(|(i, _)| i).collect()
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    use rand::SeedableRng;

    #[test]
    fn test_sample_amount() {
        let mut rng: rand_chacha::ChaCha8Rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
        let points = generate_clustered_points(
            & mut rng,
            23,
            &vec![
                ORIGIN,
                Point {
                    x1: -15.3,
                    x2: -1.5,
                },
                Point { x1: 5.0, x2: 3.0 },
            ],
            &vec![(1.0, 1.0), (2.0, 2.5), (1.4, 10.0)],
        );
        assert_eq!(points.len(), 23);
    }

    #[test]
    fn test_invalid_k() {
        let mut rng: rand_chacha::ChaCha8Rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
        let mut kmeans = KMeans {k: 100, rng: &mut rng};
        match kmeans.train(&vec![ORIGIN], 1e-3, 20) {
            Err(error::KMeansError::InvalidK) => assert!(true),
            _ => assert!(false)
        }
    }

    #[test]
    fn test_init_centroids() {
        let mut rng: rand_chacha::ChaCha8Rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);

        let points = generate_clustered_points(
            & mut rng,
            123,
            &vec![
                ORIGIN,
                Point {
                    x1: -15.3,
                    x2: -1.5,
                },
                Point { x1: 5.0, x2: 3.0 },
            ],
            &vec![(1.0, 1.0), (2.0, 2.5), (1.4, 10.0)],
        );

        let mut kmeans = KMeans {k: 100, rng: &mut rng};

        let centroids = kmeans.init_centroids(&points);

        assert_eq!(100, centroids.len());
    }

    #[test]
    fn test_assign_clusters() {
        let mut rng: rand_chacha::ChaCha8Rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);

        let centroids = &vec![
            ORIGIN,
            Point {
                x1: -15.3,
                x2: -1.5,
            },
            Point { x1: 5.0, x2: 3.0 },
        ];

        let points = generate_clustered_points(
            & mut rng,
            10,
            centroids,
            &vec![(1.0, 1.0), (2.0, 2.5), (1.4, 10.0)],
        );

        let clusters: Vec<Vec<Point>> = KMeans::<rand_chacha::ChaCha8Rng>::assign_points_to_clusters(&centroids, &points);

        let cs: Vec<usize> = clusters.iter().map(|v| v.len()).collect();
        assert_eq!(vec![5usize, 3, 2], cs);
    }

}
