use ndarray::parallel::prelude::*;
use ndarray::{Array1, Axis};
use ndarray_linalg::Norm;
use rand::prelude::*;

pub struct Cluster {
    pub centroid: ndarray::Array1<f64>,
    n: usize,
}

impl Cluster {
    fn new(n_cols: usize) -> Cluster {
        Cluster {
            centroid: ndarray::Array1::zeros(n_cols),
            n: 0,
        }
    }

    //add element to cluster
    fn add_element(&mut self, element: &ndarray::ArrayView1<f64>) {
        self.centroid += element;
        self.n += 1;
    }

    // update centroid
    fn update_centroid(&mut self) {
        if self.n != 0 {
            self.centroid /= self.n as f64;
            self.n = 0;
        }
    }
}

pub struct KMeans {
    pub n_clusters: usize,
    max_iters: usize,
    pub centroids: Vec<Cluster>,
}

impl KMeans {
    pub fn new(n_clusters: usize, max_iters: usize) -> KMeans {
        KMeans {
            n_clusters: n_clusters,
            max_iters: max_iters,
            centroids: vec![],
        }
    }

    pub fn fit(&mut self, data: ndarray::ArrayView2<f64>) -> Array1<usize> {
        let inits = data
            .rows()
            .into_iter()
            .choose_multiple(&mut rand::thread_rng(), self.n_clusters);

        for init in inits.iter() {
            let mut c = Cluster::new(data.ncols());
            c.add_element(&init);
            self.centroids.push(c);
        }
        let mut c_indexer = self.predict(data);

        for _ in 0..self.max_iters {
            self.update_centroids(data, c_indexer.view());
            let c_indexer_new = self.predict(data);
            if c_indexer_new == c_indexer {
                break;
            } else {
                c_indexer = c_indexer_new;
            }
        }
        c_indexer
    }

    pub fn predict(&self, data: ndarray::ArrayView2<f64>) -> ndarray::Array1<usize> {
        data.axis_iter(Axis(0))
            .into_par_iter()
            .map(|row| self.get_nearest(row))
            .collect::<Vec<usize>>()
            .into()
    }

    fn get_nearest(&self, values: ndarray::ArrayView1<f64>) -> usize {
        let mut min_dist = std::f64::MAX;
        let mut min_idx = 0;
        for (i, c) in self.centroids.iter().enumerate() {
            let dist = (&values - &c.centroid).norm();
            if dist < min_dist {
                min_dist = dist;
                min_idx = i;
            }
        }
        min_idx
    }

    fn update_centroids(
        &mut self,
        data: ndarray::ArrayView2<f64>,
        c_indexer: ndarray::ArrayView1<usize>,
    ) {
        for i in 0..self.n_clusters {
            self.centroids[i] = Cluster::new(data.ncols());
        }
        for (&c_ix, row) in c_indexer.iter().zip(data.rows()) {
            self.centroids[c_ix].add_element(&row);
        }
        for c_ix in 0..self.n_clusters {
            self.centroids[c_ix].update_centroid();
        }
    }
}
