from goes2go import GOES

# Band 13 - Clean IR Longwave
g13 = GOES(satellite=16, product="ABI-L1b-Rad", domain="C", bands=13)
g13.timerange(start="2024-06-01 00:00", end="2024-06-02 00:00", save_dir="./raw_data/band13")

# Band 9 - Water Vapor
g9 = GOES(satellite=16, product="ABI-L1b-Rad", domain="C", bands=9)
g9.timerange(start="2024-06-01 00:00", end="2024-06-02 00:00", save_dir="./raw_data/band9")