///////////////////////////////////
// Part I: Random number generation
///////////////////////////////////

typedef uint rng_state_t[2];

inline uint rotl(uint x, int k) {
    return (x << k) | (x >> (32 - k));
}

uint xoroshiro64star_next(rng_state_t s) {
    const uint s0 = s[0];
    uint s1 = s[1];
    const uint r = s0 * 0x9E3779BB;
    s1 ^= s0;
    s[0] = rotl(s0, 26) ^ s1 ^ (s1 << 9);
    s[1] = rotl(s1, 13);
    return r;
}

float rng_next_float(rng_state_t s) {
    return xoroshiro64star_next(s) / 4294967295.0f;
}

void rng_init(rng_state_t s, uint i, uint seed) {
    // Some prime number wizardry to initialize the random generators per iteration
    s[0] = (i + 11) * 313 + 887 + seed;
    s[1] = (i + 37) * 199 + 1129 + seed;
}

// Box Muller method to generate two std. norm. numbers using two uniformly [0, 1] distributed numbers.
float2 box_muller(rng_state_t s) {
    float u1 = rng_next_float(s);
    float u2 = rng_next_float(s);
    float r = sqrt(-2 * log(u1));
    float z0 = r * cospi(2 * u2);
    float z1 = r * sinpi(2 * u2);
    return (float2) (z0, z1);
}

//////////////////////////////////
// Part II: Monte Carlo simulation
//////////////////////////////////

typedef struct {
    int region;
    float ead;
    float lgd;
    float pd;
    float alpha;
    float treshold;
    float gamma;
    int pad;
} loan_t;

typedef float3 mat3x3_t[3];

__kernel void simulation(
    __global loan_t *restrict loans,
    __global double *restrict losses,
    __constant mat3x3_t lower,
    unsigned num_loans,
    unsigned seed
) {
    const unsigned id = get_global_id(0);

    rng_state_t s;
    rng_init(s, id, seed);

    // Sample a MVN distribution using the lower Cholesky matrix
    float2 z1 = box_muller(s);
    float2 z2 = box_muller(s);
    float3 v = (float3) (z1.x, z1.y, z2.x);
    const float risk[] = { dot(v, lower[0]), dot(v, lower[1]), dot(v, lower[2]) };

    // The loop is unrolled, because Box Muller always produces two std. norm. numbers
    double a1 = 0.0;
    double a2 = 0.0;

    for(unsigned i = 0; i < num_loans; i += 2) {
	    // The actual Monte Carlo simulation
        float2 n = box_muller(s);

        loan_t l1 = loans[i];
        loan_t l2 = loans[i + 1];

	    float v1 = l1.alpha * risk[l1.region] + l1.gamma * n.x;
	    a1 += select(0.0f, l1.ead * l1.lgd, v1 < l1.treshold);

        float v2 = l2.alpha * risk[l2.region] + l2.gamma * n.y;
	    a2 += select(0.0f, l2.ead * l2.lgd, v2 < l2.treshold);
    }

    losses[id] = a1 + a2;
}
