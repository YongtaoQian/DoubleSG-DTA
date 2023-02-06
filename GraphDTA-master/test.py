from rdkit import Chem
m = Chem.MolFromSmiles('C1OC1')
print('\t'.join(['id', 'num', 'symbol', 'degree', 'charge', 'hybrid']))
for atom in m.GetAtoms():
    print(atom.GetIdx(), end='\t')
    print(atom.GetAtomicNum(), end='\t')
    print(atom.GetSymbol(), end='\t')
    print(atom.GetDegree(), end='\t')
    print(atom.GetFormalCharge(), end='\t')
    print(atom.GetHybridization(), end='\t')
    print(atom.GetExplicitValence(), end='\t')
    print(atom.GetImplicitValence(), end='\t')
    print(atom.GetTotalValence())
