"""Tests for Architect."""
from src.core import Architect
def test_init(): assert Architect().get_stats()["ops"] == 0
def test_op(): c = Architect(); c.generate(x=1); assert c.get_stats()["ops"] == 1
def test_multi(): c = Architect(); [c.generate() for _ in range(5)]; assert c.get_stats()["ops"] == 5
def test_reset(): c = Architect(); c.generate(); c.reset(); assert c.get_stats()["ops"] == 0
def test_service_name(): c = Architect(); r = c.generate(); assert r["service"] == "architect"
