use criterion::{black_box, criterion_group, criterion_main, Criterion};
use position_aware_embed::{position_aware_embed, hash_embed, VectorIndex};

fn bench_embed(c: &mut Criterion) {
    c.bench_function("position_aware_embed", |b| {
        b.iter(|| position_aware_embed(black_box("check disk usage"), black_box(64)))
    });
    c.bench_function("hash_embed", |b| {
        b.iter(|| hash_embed(black_box("check disk usage"), black_box(64)))
    });
}

fn bench_search(c: &mut Criterion) {
    let mut idx = VectorIndex::new(64);
    for i in 0..200 {
        idx.add(&format!("command {}", i), &format!("cmd_{}", i));
    }
    c.bench_function("search_200_vectors", |b| {
        b.iter(|| idx.search(black_box("how much disk space"), black_box(5)))
    });
}

criterion_group!(benches, bench_embed, bench_search);
criterion_main!(benches);
