import sys
sys.path.insert(0, '../')
from friedrich.mtwilson import parse_mwo_group_spot


table = parse_mwo_group_spot('/local/tmp/Mt_Wilson_Tilt')

print(len(table))