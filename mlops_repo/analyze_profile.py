import pstats

# Load the profiling results
p = pstats.Stats('output.prof')

# Strip directory paths, sort by cumulative time, and print the top 10 functions
p.strip_dirs().sort_stats('cumtime').print_stats(10)