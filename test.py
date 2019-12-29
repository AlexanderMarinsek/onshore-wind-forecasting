from datetime import datetime

day_of_year1 = datetime.now().timetuple().tm_yday
day_of_year2 = datetime(2003, 1, 2).timetuple().tm_yday

a = 1