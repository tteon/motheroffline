mod vector;
mod index;
mod distance;

use index::VectorIndex;
use vector::Vector;

fn main() {
    let mut index = VectorIndex::new();

    // Insert sample vectors
    index.insert(Vector::new(0, vec![1.0, 2.0, 3.0]));
    index.insert(Vector::new(1, vec![2.0, 3.0, 4.0]));
    index.insert(Vector::new(2, vec![4.0, 5.0, 6.0]));

    // Query
    let query = vec![1.5, 2.5, 3.5];
    let results = index.search(&query, 2);

    println!("Top-k results:");
    for (id, distance) in results {
        println!("ID: {}, Distance: {:.4}", id, distance);
    }
}
