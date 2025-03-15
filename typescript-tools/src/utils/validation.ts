/**
 * Validates that the input is a non-empty array of valid numbers.
 * 
 * @param numbers The array to validate
 * @throws Error if validation fails
 */
export function validateNumbers(numbers: number[]): void {
    if (!Array.isArray(numbers) || numbers.length === 0) {
        throw new Error("Input must be a non-empty array of numbers.");
    }
    for (const num of numbers) {
        if (typeof num !== 'number' || isNaN(num)) {
            throw new Error("All items in the array must be valid numbers.");
        }
    }
}

/**
 * Validates that the input is a valid matrix (2D array of numbers).
 * 
 * @param matrix The matrix to validate
 * @throws Error if validation fails
 */
export function validateMatrix(matrix: number[][]): void {
    if (!Array.isArray(matrix) || matrix.length === 0) {
        throw new Error("Input must be a non-empty 2D array.");
    }
    
    const rowLength = matrix[0].length;
    for (const row of matrix) {
        if (!Array.isArray(row) || row.length !== rowLength) {
            throw new Error("All rows must be arrays of the same length.");
        }
        
        for (const value of row) {
            if (typeof value !== 'number' || isNaN(value)) {
                throw new Error("All values in the matrix must be valid numbers.");
            }
        }
    }
}

/**
 * Types of statistical calculations that can be validated
 */
export type CalculationType = 
    | "mean" 
    | "median" 
    | "mode" 
    | "standardDeviation" 
    | "variance" 
    | "probabilityDistribution" 
    | "eigenvalues" 
    | "eigenvectors";

/**
 * Result of a calculation validation
 */
export interface ValidationResult {
    isValid: boolean;
    expectedResult: number | number[] | number[][] | { [key: string]: number };
    calculatedResult: number | number[] | number[][] | { [key: string]: number };
    error?: string;
}

/**
 * Validates the correctness of statistical calculator results.
 * 
 * @param input Input data used for calculation (array or matrix)
 * @param calculatedResult The result to validate 
 * @param calculationType Type of calculation to verify
 * @param options Additional options needed for specific calculations
 * @returns A validation result object
 */
export function validateCalculation(
    input: number[] | number[][],
    calculatedResult: any,
    calculationType: CalculationType,
    options?: {
        populationStdDev?: boolean; // True for population, false for sample
        precision?: number; // Decimal precision for comparison
    }
): ValidationResult {
    try {
        const precision = options?.precision ?? 10; // Default to 10 decimal places
        const epsilon = Math.pow(10, -precision);
        
        let expectedResult: any;
        
        // Calculate the expected result based on calculator type
        switch (calculationType) {
            case "mean":
                validateNumbers(input as number[]);
                expectedResult = calculateCorrectMean(input as number[]);
                break;
                
            case "median":
                validateNumbers(input as number[]);
                expectedResult = calculateCorrectMedian(input as number[]);
                break;
                
            case "mode":
                validateNumbers(input as number[]);
                expectedResult = calculateCorrectMode(input as number[]);
                break;
                
            case "standardDeviation":
                validateNumbers(input as number[]);
                expectedResult = calculateCorrectStandardDeviation(
                    input as number[], 
                    options?.populationStdDev ?? false
                );
                break;
                
            case "variance":
                validateNumbers(input as number[]);
                expectedResult = calculateCorrectVariance(
                    input as number[],
                    options?.populationStdDev ?? false
                );
                break;
                
            case "probabilityDistribution":
                validateNumbers(input as number[]);
                expectedResult = calculateCorrectProbabilityDistribution(input as number[]);
                break;
                
            case "eigenvalues":
                validateMatrix(input as number[][]);
                expectedResult = calculateCorrectEigenvalues(input as number[][]);
                break;
                
            case "eigenvectors":
                validateMatrix(input as number[][]);
                expectedResult = calculateCorrectEigenvectors(input as number[][]);
                break;
                
            default:
                throw new Error(`Unknown calculator type: ${calculationType}`);
        }
        
        // Compare calculated result with expected result based on result type
        let isValid = false;
        
        if (Array.isArray(expectedResult) && Array.isArray(calculatedResult)) {
            // For array results (mode, probability distribution, eigenvalues, eigenvectors)
            if (expectedResult.length === 0 && calculatedResult.length === 0) {
                isValid = true;
            } else if (Array.isArray(expectedResult[0]) && Array.isArray(calculatedResult[0])) {
                // 2D array (matrix) comparison for eigenvectors
                isValid = matricesEqual(expectedResult, calculatedResult, epsilon);
            } else {
                // 1D array comparison for mode, eigenvalues
                isValid = arraysEqual(expectedResult, calculatedResult, epsilon);
            }
        } else if (typeof expectedResult === 'object' && typeof calculatedResult === 'object') {
            // For object results (probability distribution)
            isValid = objectsEqual(expectedResult, calculatedResult, epsilon);
        } else if (typeof expectedResult === 'number' && typeof calculatedResult === 'number') {
            // For numeric results (mean, median, standard deviation, variance)
            isValid = Math.abs(expectedResult - calculatedResult) < epsilon;
        }
        
        return {
            isValid,
            expectedResult,
            calculatedResult,
            error: isValid ? undefined : `Incorrect ${calculationType} calculation.`
        };
    } catch (error) {
        return {
            isValid: false,
            expectedResult: typeof input === 'object' && Array.isArray(input[0]) ? [] : 0,
            calculatedResult,
            error: error instanceof Error ? error.message : String(error)
        };
    }
}

// -------- Correct implementations of statistical calculations --------

/**
 * Calculate the correct mean without any errors
 */
function calculateCorrectMean(numbers: number[]): number {
    const sum = numbers.reduce((acc, num) => acc + num, 0);
    return sum / numbers.length;
}

/**
 * Calculate the correct median without any errors
 */
function calculateCorrectMedian(numbers: number[]): number {
    const sorted = [...numbers].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    if (sorted.length % 2 === 0) {
        return (sorted[mid - 1] + sorted[mid]) / 2;
    } else {
        return sorted[mid];
    }
}

/**
 * Calculate the correct mode (most frequent values) without any errors
 */
function calculateCorrectMode(numbers: number[]): number[] {
    const frequency: Record<number, number> = {};
    let maxFrequency = 0;
    
    // Count frequency of each number
    for (const num of numbers) {
        frequency[num] = (frequency[num] || 0) + 1;
        maxFrequency = Math.max(maxFrequency, frequency[num]);
    }
    
    // Find all numbers with the maximum frequency
    const modes: number[] = [];
    for (const num in frequency) {
        if (frequency[num] === maxFrequency) {
            modes.push(Number(num));
        }
    }
    
    return modes.sort((a, b) => a - b); // Sort for consistent order
}

/**
 * Calculate the correct standard deviation without any errors
 */
function calculateCorrectStandardDeviation(numbers: number[], isPopulation: boolean = false): number {
    const mean = calculateCorrectMean(numbers);
    const squaredDifferences = numbers.map(num => Math.pow(num - mean, 2));
    const sumSquaredDiff = squaredDifferences.reduce((acc, val) => acc + val, 0);
    
    // Use N for population, N-1 for sample
    const divisor = isPopulation ? numbers.length : numbers.length - 1;
    
    return Math.sqrt(sumSquaredDiff / divisor);
}

/**
 * Calculate the correct variance without any errors
 */
function calculateCorrectVariance(numbers: number[], isPopulation: boolean = false): number {
    const stdDev = calculateCorrectStandardDeviation(numbers, isPopulation);
    return Math.pow(stdDev, 2);
}

/**
 * Calculate the correct probability distribution
 */
function calculateCorrectProbabilityDistribution(numbers: number[]): { [key: string]: number } {
    const frequency: Record<number, number> = {};
    const total = numbers.length;
    
    // Count occurrences of each value
    for (const num of numbers) {
        frequency[num] = (frequency[num] || 0) + 1;
    }
    
    // Convert counts to probabilities
    const distribution: Record<string, number> = {};
    for (const num in frequency) {
        distribution[num] = frequency[num] / total;
    }
    
    return distribution;
}

/**
 * Calculate eigenvalues of a matrix using the power method
 * Note: This is a simplified implementation for common cases
 * For a complete implementation, use a numerical library
 */
function calculateCorrectEigenvalues(matrix: number[][]): number[] {
    // This is a placeholder - full eigenvalue calculation 
    // requires complex numerical methods beyond the scope of this utility
    // In a real implementation, use a math library like mathjs
    
    // For 2x2 matrices, we can use the quadratic formula
    if (matrix.length === 2 && matrix[0].length === 2) {
        const a = 1;
        const b = -(matrix[0][0] + matrix[1][1]);
        const c = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
        
        const discriminant = b * b - 4 * a * c;
        if (discriminant >= 0) {
            const sqrtDiscriminant = Math.sqrt(discriminant);
            const lambda1 = (-b + sqrtDiscriminant) / (2 * a);
            const lambda2 = (-b - sqrtDiscriminant) / (2 * a);
            return [lambda1, lambda2].sort((a, b) => b - a); // Sort in descending order
        }
    }
    
    // For other cases, return an empty array
    // In a real implementation, you would use a numerical method library
    return [];
}

/**
 * Calculate eigenvectors of a matrix
 * Note: This is a simplified implementation for common cases
 * For a complete implementation, use a numerical library
 */
function calculateCorrectEigenvectors(matrix: number[][]): number[][] {
    // This is a placeholder - full eigenvector calculation 
    // requires complex numerical methods beyond the scope of this utility
    // In a real implementation, use a math library like mathjs
    
    // For 2x2 matrices with real eigenvalues, we can directly compute eigenvectors
    if (matrix.length === 2 && matrix[0].length === 2) {
        const eigenvalues = calculateCorrectEigenvalues(matrix);
        
        if (eigenvalues.length === 2) {
            const eigenvectors: number[][] = [];
            
            for (const lambda of eigenvalues) {
                // For each eigenvalue λ, find a non-zero solution to (A - λI)v = 0
                const a11 = matrix[0][0] - lambda;
                const a12 = matrix[0][1];
                const a21 = matrix[1][0];
                const a22 = matrix[1][1] - lambda;
                
                // Try to solve for eigenvector
                if (Math.abs(a11) > Math.abs(a21)) {
                    // Use first row
                    if (a11 !== 0) {
                        const v2 = 1;
                        const v1 = -a12 * v2 / a11;
                        const magnitude = Math.sqrt(v1 * v1 + v2 * v2);
                        eigenvectors.push([v1 / magnitude, v2 / magnitude]);
                    } else if (a12 !== 0) {
                        eigenvectors.push([1, 0]);
                    }
                } else {
                    // Use second row
                    if (a21 !== 0) {
                        const v2 = 1;
                        const v1 = -a22 * v2 / a21;
                        const magnitude = Math.sqrt(v1 * v1 + v2 * v2);
                        eigenvectors.push([v1 / magnitude, v2 / magnitude]);
                    } else if (a22 !== 0) {
                        eigenvectors.push([0, 1]);
                    }
                }
            }
            
            return eigenvectors;
        }
    }
    
    // For other cases, return an empty array
    // In a real implementation, you would use a numerical method library
    return [];
}

// -------- Helper comparison functions --------

/**
 * Helper to check if two arrays have approximately equal values
 */
function arraysEqual(a: number[], b: number[], epsilon: number = 1e-10): boolean {
    if (a.length !== b.length) return false;
    
    // Sort both arrays for value comparison (ignores order)
    const sortedA = [...a].sort((x, y) => x - y);
    const sortedB = [...b].sort((x, y) => x - y);
    
    for (let i = 0; i < sortedA.length; i++) {
        if (Math.abs(sortedA[i] - sortedB[i]) > epsilon) return false;
    }
    
    return true;
}

/**
 * Helper to check if two matrices have approximately equal values
 */
function matricesEqual(a: number[][], b: number[][], epsilon: number = 1e-10): boolean {
    if (a.length !== b.length) return false;
    
    for (let i = 0; i < a.length; i++) {
        if (a[i].length !== b[i].length) return false;
        
        for (let j = 0; j < a[i].length; j++) {
            if (Math.abs(a[i][j] - b[i][j]) > epsilon) return false;
        }
    }
    
    return true;
}

/**
 * Helper to check if two objects have approximately equal values
 */
function objectsEqual(a: { [key: string]: number }, b: { [key: string]: number }, epsilon: number = 1e-10): boolean {
    const keysA = Object.keys(a).sort();
    const keysB = Object.keys(b).sort();
    
    if (keysA.length !== keysB.length) return false;
    
    for (let i = 0; i < keysA.length; i++) {
        if (keysA[i] !== keysB[i]) return false;
        if (Math.abs(a[keysA[i]] - b[keysB[i]]) > epsilon) return false;
    }
    
    return true;
}