new_lines = []
with open('4f5s.pdb') as f:
    g = f.read().splitlines()
    for line in [i for i in g if not i.startswith('ANISOU')]:
        new_lines.append(line)
    f.close()
with open('4f5s.pdb','w+') as f:
    for line in new_lines:
        f.write("%s\n" % line)
    f.close()
