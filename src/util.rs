use std::cmp::Ordering;

///
/// Because there is no standard Ordering for f32
/// TODO: probably add a sigma tolerance to test against
/// 
pub fn cmp_f32(a: f32, b: f32) -> Ordering {
    if a < b {
        Ordering::Less
    } else if a > b {
        Ordering::Greater
    } else {
        Ordering::Equal
    }
}
