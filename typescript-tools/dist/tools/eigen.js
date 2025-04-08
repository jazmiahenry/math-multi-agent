"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.calculateEigen = void 0;
/**
 * Calculates eigenvalues and eigenvectors for a 2x2 matrix.
 *
 * @param matrix A 2x2 matrix represented as an array of arrays
 * @returns Object containing eigenvalues and eigenvectors
 * @throws Error if input is not a 2x2 matrix
 */
function calculateEigen(matrix) {
    if (!Array.isArray(matrix) || matrix.length !== 2 || !Array.isArray(matrix[0]) || matrix[0].length !== 2) {
        throw new Error("Only 2x2 matrices are supported.");
    }
    const a = matrix[0][0];
    const b = matrix[0][1];
    const c = matrix[1][0];
    const d = matrix[1][1];
    // Calculate eigenvalues using the characteristic equation
    const trace = a + d;
    const determinant = a * d - b * c;
    const discriminant = Math.sqrt(trace * trace - 4 * determinant);
    const eigenvalue1 = (trace + discriminant) / 2;
    const eigenvalue2 = (trace - discriminant) / 2;
    const eigenvalues = [eigenvalue1, eigenvalue2];
    // Calculate eigenvectors for each eigenvalue
    const eigenvectors = eigenvalues.map(lambda => {
        if (b !== 0) {
            const vec = [1, (lambda - a) / b];
            return normalizeVector(vec);
        }
        else if (c !== 0) {
            const vec = [(lambda - d) / c, 1];
            return normalizeVector(vec);
        }
        else {
            // Diagonal matrix: return the standard basis vector
            return a === lambda ? [1, 0] : [0, 1];
        }
    });
    return { eigenvalues, eigenvectors };
}
exports.calculateEigen = calculateEigen;
/**
 * Normalizes a vector to unit length
 *
 * @param vec Vector to normalize
 * @returns Normalized vector with the same direction but unit length
 */
function normalizeVector(vec) {
    const norm = Math.sqrt(vec.reduce((acc, val) => acc + val * val, 0));
    return vec.map(val => val / norm);
}
