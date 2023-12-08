use crate::kmeans;
use ndarray::parallel::prelude::*;
use ndarray::prelude::*;
use ndarray_linalg::Norm;
use rand::prelude::*;
use rayon::prelude::*;

pub fn train(
    data: ArrayView2<f64>,
    n_subspaces: usize,
    n_clusters: usize,
    max_iters: usize,
) -> (Array3<f64>, Vec<Vec<usize>>) {
    let n_data = data.nrows();
    let n_dim_data = data.ncols();
    let n_dim_subspace = n_dim_data / n_subspaces;
    let mut codebooks = ndarray::Array3::<f64>::zeros((n_subspaces, n_clusters, n_dim_subspace));
    let mut subspaces = Vec::new();
    for _ in 0..n_subspaces {
        let mut rng = rand::thread_rng();
        let subspace_indices = (0..n_dim_data).choose_multiple(&mut rng, n_dim_subspace);
        subspaces.push(subspace_indices);
    }
    for (subspace_i, subspace_indices) in subspaces.iter().enumerate() {
        let mut sub_data = Array2::<f64>::zeros((n_data, n_dim_subspace));
        for (i, row) in data.outer_iter().enumerate() {
            // Slice row with subspace indices
            for (j, index) in subspace_indices.iter().enumerate() {
                sub_data[[i, j]] = row[*index];
            }
        }
        let mut logic = kmeans::KMeans::new(n_clusters, max_iters);
        logic.fit(sub_data.view());
        let mut codebook = Array2::<f64>::zeros((n_clusters, n_dim_subspace));
        for (i, cluster) in logic.centroids.iter().enumerate() {
            for (j, &val) in cluster.centroid.iter().enumerate() {
                codebook[[i, j]] = val
            }
        }
        codebooks
            .slice_mut(s![subspace_i, .., ..])
            .assign(&codebook);
    }
    (codebooks, subspaces)
}

pub fn encode(
    codebooks: ArrayView3<f64>,
    subspaces: &Vec<Vec<usize>>,
    data: ArrayView2<f64>,
) -> Array2<usize> {
    let n_data = data.nrows();
    let n_subspaces = subspaces.len();
    let n_dim_subspace = subspaces[0].len();
    let mut pqcode = Array2::<usize>::zeros((n_data, n_subspaces));
    for (subspace_i, subspace_indices) in subspaces.iter().enumerate() {
        let mut sub_data = Array2::<f64>::zeros((n_data, n_dim_subspace));
        for (i, row) in data.outer_iter().enumerate() {
            // Slice row with subspace indices
            for (j, index) in subspace_indices.iter().enumerate() {
                sub_data[[i, j]] = row[*index];
            }
        }
        let mut sub_pqcode = Array1::<usize>::zeros(n_data);
        let min_index = sub_data
            .axis_iter(Axis(0))
            .enumerate()
            .par_bridge()
            .map(|(_, row)| {
                let mut min_dist = std::f64::MAX;
                let mut min_index = 0;
                for (j, centroid) in codebooks
                    .slice(s![subspace_i, .., ..])
                    .outer_iter()
                    .enumerate()
                {
                    let dist = (&row - &centroid).norm();
                    if dist < min_dist {
                        min_dist = dist;
                        min_index = j;
                    }
                }
                min_index
            })
            .collect::<Vec<_>>();
        min_index
            .iter()
            .enumerate()
            .for_each(|(i, &x)| sub_pqcode[i] = x);

        pqcode.slice_mut(s![.., subspace_i]).assign(&sub_pqcode);
    }
    pqcode
}

pub fn search(
    codebooks: ArrayView3<f64>,
    subspaces: &Vec<Vec<usize>>,
    pqcode: ArrayView2<usize>,
    query: ArrayView1<f64>,
) -> Vec<f64> {
    let n_subspaces = subspaces.len();
    let n_dim_subspace = subspaces[0].len();
    let mut all_distances = Vec::new();
    let n_clusters = codebooks.shape()[1];
    let mut dist_table = Array2::<f64>::zeros((n_subspaces, n_clusters));
    for (subspace_i, subspace_indices) in subspaces.iter().enumerate() {
        let sub_codebooks = codebooks.slice(s![subspace_i, .., ..]);
        let mut sub_query = Array1::<f64>::zeros(n_dim_subspace);
        for (i, index) in subspace_indices.iter().enumerate() {
            sub_query[i] = query[*index];
        }
        let dists = Array1::from(
            (&sub_codebooks - &sub_query)
                .outer_iter()
                .map(|x| x.norm())
                .collect::<Vec<_>>(),
        );
        dist_table.slice_mut(s![subspace_i, ..]).assign(&dists);
    }
    for row in pqcode.axis_iter(Axis(0)) {
        let mut dist = 0.0;
        for (j, &index) in row.iter().enumerate() {
            dist += dist_table[[j, index]];
        }
        all_distances.push(dist);
    }
    all_distances
}
