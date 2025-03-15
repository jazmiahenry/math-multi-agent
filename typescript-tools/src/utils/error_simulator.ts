/**
 * Simulates errors by randomly throwing errors or returning incorrect results.
 * 
 * @param fn Function that produces the correct result
 * @param errorConfig Configuration for error simulation
 * @returns Either the correct result or an incorrect one
 */
export interface ErrorSimulatorConfig<T> {
  /** Probability of throwing an error (0-1) */
  throwErrorRate?: number;
  /** Probability of returning incorrect result (0-1) */
  incorrectResultRate?: number;
  /** Optional function to generate incorrect results */
  incorrectResultFn?: () => T;
  /** Whether to log errors silently without breaking execution */
  silentError?: boolean;
}

export function simulateError<T>(
  fn: () => T, 
  errorConfig: ErrorSimulatorConfig<T> = {}
): T {
  const {
    throwErrorRate = 0.2,
    incorrectResultRate = 0.2,
    incorrectResultFn,
    silentError = false
  } = errorConfig;

  // Total error rate is the sum of both types of errors
  const totalErrorRate = throwErrorRate + incorrectResultRate;
  
  // Validate error rates
  if (totalErrorRate < 0 || totalErrorRate > 1) {
    throw new Error("Combined error rate must be between 0 and 1");
  }

  const random = Math.random();
  
  // Case 1: Throw an error
  if (random < throwErrorRate) {
    const error = new Error("Simulated error");
    if (silentError) {
      console.error("[Silent Error]", error);
    } else {
      throw error;
    }
  }
  
  // Case 2: Return incorrect result
  if (random >= throwErrorRate && random < totalErrorRate) {
    if (incorrectResultFn) {
      return incorrectResultFn();
    }
    
    // Default incorrect result handling based on type
    const correctResult = fn();
    const type = typeof correctResult;
    
    switch (type) {
      case 'number':
        // For numbers, add/subtract a random value
        return (correctResult as any * (Math.random() > 0.5 ? 0.5 : 1.5)) as T;
      case 'string':
        // For strings, scramble characters
        const str = correctResult as string;
        return (str.split('').sort(() => Math.random() - 0.5).join('')) as unknown as T;
      case 'boolean':
        // For booleans, flip the value
        return (!correctResult) as unknown as T;
      case 'object':
        if (Array.isArray(correctResult)) {
          // For arrays, shuffle elements
          return ([...correctResult].sort(() => Math.random() - 0.5)) as unknown as T;
        }
        // For objects, can't easily generate "wrong" version without more context
        return correctResult;
      default:
        return correctResult;
    }
  }
  
  // Case 3: Return correct result
  return fn();
}