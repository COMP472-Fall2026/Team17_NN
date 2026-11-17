"""Tiny tests that detect fundamental errors without external classifiers."""
import unittest
import numpy as np
from models.decision_tree import DecisionTree
from models.random_forest import RandomForest

class TreeForestTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(7)
        x=rng.normal(size=(180,4)).astype('float32')
        y=np.where(x[:,0]>0.7,2,np.where(x[:,0]<-0.7,0,1)).astype(int)
        self.x,self.y=x,y

    def test_tree_learns_multiclass(self):
        m=DecisionTree(max_depth=5,min_samples_leaf=2,max_features=None).fit(self.x,self.y)
        self.assertGreater(np.mean(m.predict(self.x)==self.y),0.95)
        self.assertTrue(np.allclose(m.predict_proba(self.x).sum(1),1))

    def test_forest_learns_multiclass(self):
        m=RandomForest(n_estimators=8,max_depth=7,min_samples_leaf=2).fit(self.x,self.y)
        self.assertGreater(np.mean(m.predict(self.x)==self.y),0.90)
        self.assertTrue(np.allclose(m.predict_proba(self.x).sum(1),1))

    def test_determinism(self):
        x,y=self.x,self.y
        a=RandomForest(n_estimators=5,random_state=3).fit(x,y).predict(x)
        b=RandomForest(n_estimators=5,random_state=3).fit(x,y).predict(x)
        np.testing.assert_array_equal(a,b)

    def test_nonconsecutive_labels(self):
        y=np.where(self.y==0,10,np.where(self.y==1,30,90))
        m=RandomForest(n_estimators=5).fit(self.x,y)
        self.assertTrue(set(np.unique(m.predict(self.x))).issubset({10,30,90}))

    def test_constant_feature(self):
        x=np.ones((40,3),dtype=np.float32)
        y=np.array([1,2]*20)
        m=DecisionTree().fit(x,y)
        self.assertEqual(m.predict(x).shape,(40,))

if __name__=='__main__':unittest.main()
