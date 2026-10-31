"""Regression cases for failure modes missed by the original station-only check."""
import copy
import unittest

from w2lib.drivability import flight_landing, trap_search


class DrivabilityTests(unittest.TestCase):
    def trap(self):
        return dict(cycle=4, width_at=14, warn_lead_m=20,
                    stations=[dict(off=10,bands=[dict(d0=-7,d1=7,on=(0,4),phase=0,len=6)])])

    def test_permanent_full_block_is_rejected(self):
        r=trap_search(self.trap(),20)
        self.assertEqual(r["failed"],r["phases"])

    def test_inactive_bands_leave_the_road_open(self):
        tr=self.trap();tr["stations"][0]["bands"][0]["on"]=(0,0)
        # Remove rather than claim an empty ON window still has a tolerance pad.
        tr["stations"][0]["bands"]=[]
        tr["stations"].append(dict(off=20,bands=[dict(d0=20,d1=21,on=(0,4),len=6)]))
        tr["stations"]=tr["stations"][1:]
        self.assertEqual(trap_search(tr,20)["failed"],0)

    def test_finite_hazard_length_catches_between_centre_activation(self):
        tr=self.trap();tr["cycle"]=10
        # At t=1.5 the centre is inactive; the front enters at t=1.28.
        tr["stations"][0]["bands"][0]["on"]=(1.25,1.35)
        r=trap_search(tr,20,phase_step=10)
        self.assertEqual(r["failed"],1)

    def test_witness_obeys_velocity_and_acceleration(self):
        tr=self.trap();tr["stations"][0]["bands"][0].update(d0=-7,d1=-1,on=(0,4))
        r=trap_search(tr,20,store_witness=True)
        d=r["witness"]["d"]
        self.assertLessEqual(max(abs(b-a)/.1 for a,b in zip(d,d[1:])),2.8+1e-8)
        self.assertLessEqual(max(abs(c-2*b+a)/.1**2 for a,b,c in zip(d,d[1:],d[2:])),7+1e-8)
        self.assertAlmostEqual(d[0],0)

    def test_actual_gravity_rejects_normal_hop_over_36m(self):
        j=dict(gap=36,landing_len=36,z=dict(lip=25.73,land0=24.13,land1=20.03))
        self.assertFalse(flight_landing(j,25,7)["success"])
        self.assertTrue(flight_landing(j,32,8.8)["success"])

    def test_landing_intersection_is_after_front_edge(self):
        j=dict(gap=36,landing_len=36,z=dict(lip=25.73,land0=24.13,land1=20.03))
        r=flight_landing(j,32,8.8)
        self.assertGreater(r["landing_inset_m"],6)
        self.assertLess(r["relative_impact_deg"],16)


if __name__=="__main__":
    unittest.main()
