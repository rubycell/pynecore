# Pine Script Examples

This directory contains 411 Pine Script examples extracted from the official documentation.

## Source

Examples were extracted from: `doc-pinescript-6/extracted_examples.json`

## Testing

These examples can be used to validate the PyneCore Pine Script interpreter:

```bash
# Test a single example
pyne pynecore/sample/pinescript/ex_001_bar_index.pine

# Test all examples
for file in pynecore/sample/pinescript/ex_*.pine; do
    echo "Testing $file..."
    pyne "$file" || echo "FAILED: $file"
done
```

## Example Categories

- **Complex**: 37 examples
- **Medium**: 216 examples
- **Simple**: 104 examples

## All comment out

We have to comment out all file with library keyword