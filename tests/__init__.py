import test_use_cluster 
import unittest

suite = unittest.TestLoader().loadTestsFromModule(test_use_cluster)
unittest.TextTestRunner(verbosity=2).run(suite)
