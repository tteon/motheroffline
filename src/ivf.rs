use crate::vector::Vector;
use crate::distance::l2_distance;
use rand::seq::SliceRandom;
use std::collections::HashMap;
use std::fs::File;
use std::error::Error;
use csv::Reader;
use rayon::prelude::*;

pub fn load_vectors_from_csv(path: &str) -> Result<Vec<Vector>, Box<dyn Error>> {
    let file = File::open(path)?;
    let mut rdr = Reader::from_reader(file);
    let mut result = Vec::new();

    for record in rdr.deserialize() {
        let vec: Vector = record?;
        result.push(vec);
    }

    Ok(result)
}

pub struct IVFIndex {
    pub centroids: Vec<Vec<f32>>, // k centroids
    pub inverted_lists: HashMap<usize, Vec<Vector>>,
    pub k: usize,
}

impl IVFIndex {
    pub fn new(k: usize) -> Self {
        Self {
            centroids: vec![],
            inverted_lists: HashMap::new(),
            k,
        }
    }

    pub fn train(&mut self, data: &[Vector]) {
        let mut rng = rand::thread_rng();
        self.centroids = data
            .choose_multiple(&mut rng, self.k)
            .map(|v| v.data.clone())
            .collect();

        for v in data {
            let mut best_cid = 0;
            let mut best_dist = f32::MAX;

            for (cid, centroid) in self.centroids.iter().enumerate() {
                let dist = l2_distance(&v.data, centroid);
                if dist < best_dist {
                    best_dist = dist;
                    best_cid = cid;
                }
            }

            self.inverted_lists.entry(best_cid).or_default().push(v.clone());
        }
    }

    /// Parallel search using rayon
    pub fn search_parallel(&self, query: &[f32], top_k: usize, nprobe: usize) -> Vec<(usize, f32)> {
        let mut centroid_dists: Vec<(usize, f32)> = self.centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, l2_distance(query, c)))
            .collect();

        centroid_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let top_centroids = &centroid_dists[..nprobe];

        let candidates: Vec<(usize, f32)> = top_centroids
            .par_iter()
            .flat_map(|(cid, _)| {
                self.inverted_lists
                    .get(cid)
                    .into_par_iter()
                    .flat_map_iter(|vectors| {
                        vectors.par_iter()
                            .map(|v| (v.id, l2_distance(&v.data, query)))
                            .collect::<Vec<_>>()
                    })
            })
            .collect();

        let mut sorted = candidates;
        sorted.par_sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        sorted.truncate(top_k);
        sorted
    }

    /// Vanilla (non-parallel) IVF search
    pub fn search_vanilla(&self, query: &[f32], top_k: usize, nprobe: usize) -> Vec<(usize, f32)> {
        let mut centroid_dists: Vec<(usize, f32)> = self.centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, l2_distance(query, c)))
            .collect();

        centroid_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let top_centroids = &centroid_dists[..nprobe];

        let mut candidates: Vec<(usize, f32)> = Vec::new();

        for (cid, _) in top_centroids {
            if let Some(vectors) = self.inverted_lists.get(cid) {
                for v in vectors {
                    let dist = l2_distance(&v.data, query);
                    candidates.push((v.id, dist));
                }
            }
        }

        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        candidates.truncate(top_k);
        candidates
    }
}
