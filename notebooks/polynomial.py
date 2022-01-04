import numpy as np
import torch

class Monomial:
    def __init__(self, indexes=[]):
        self.indexes = list(indexes)
        
    def __call__(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        y = torch.ones(x.shape[:-1] + (1,))
        for idx in self.indexes:
            y[...,0] *= x[...,idx]
        return y
    
#     def copy(self):
#         return Monomial(self.coefficient, self.indexes, self.offsets)
    
#     def __imul__(self, other):
#         if isinstance(other, (int, float)):
#             self.coefficient *= other
#         elif isinstance(other, Monomial)):
#             self.coefficient *= other.coefficient
#             self.indexes += other.indexes
#             self.offsets += other.offsets
#         else:
#             raise NotImplementedError()
        
#     def __rmul__(self, other):
#         result = self.copy()
#         result *= other
#         return result        
    
#     def add_factor(self, index, offset=None):
#         self.indexes.append(index)
#         self.offsets.append(offset if offset else None)
        
#     def pop_factor(self):
#         return self.indexes.pop(), self.offsetss.pop()
    
    @classmethod
    def from_random_process(cls, max_index, degree, sigma=1.):
        """Algorithm inspired by the chinese restaurant process."""
        assert max_index >= degree  # otherwise this function may fail unexpectedly
        all_indexes = list(range(max_index))
        np.random.shuffle(all_indexes)
        
        indexes = []
        for i in range(degree):
            new_index_prob = sigma / (sigma + i)
            if np.random.random() < new_index_prob:
                next_index = all_indexes.pop()
            else:
                next_index = np.random.choice(indexes)
            indexes.append(next_index)
        
        return cls(indexes)
    
    def degree(self):
        return len(self.indexes)
    
    def __str__(self):
        factors = []
        for idx in zip(self.indexes):
            factors.append(f"x_{{{idx}}}")
        return '*'.join(factors)
    
    def __repr__(self):
        return str(self)
    
    def latex_string(self):
        if self.degree() == 0:
            return ""
        
        coeffs, counts = np.unique(self.indexes, return_counts=True)
        factors = []
        for coef, count in zip(coeffs, counts):
            factors.append(f"x_{{{coef}}}")
            if count > 1:
                factors.append(f"^{{{count}}}")

        return ''.join(factors)
    
    def get_exponents(self, n_exponents = None):
        if not n_exponents:
            n_exponents = max(self.indexes)+1
        exponents = np.zeros(n_exponents)
        for idx in self.indexes:
            exponents[idx] += 1
        return exponents
                
    
class Polynomial:
    def __init__(self, monomials=None, coefficients=None):
        if not monomials:
            monomials = []
        
        ordering = np.argsort([m.degree() for m in monomials])
        self.monomials = [monomials[idx] for idx in ordering]
        
        if coefficients is None:
            coefficients = np.ones_like(self.monomials)
        self.coefficients = [coefficients[idx] for idx in ordering]
        
    def __call__(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        y = torch.zeros(x.shape[:-1] + (1,))
        for coef, monomial in zip(self.coefficients, self.monomials):
            y += coef * monomial(x)
        return y
    
    def add_monomial(self, monomial, coefficient=1.):
        self.monomials.append(monomial)
        self.coefficients.append(coefficient)
        
    @classmethod
    def from_random_monomials(cls, max_index, n_monomials,
                              coefficient_distribution,
                              degree_distribution,
                              sigma=1.):
        
        coefficients = [coefficient_distribution() for _ in range(n_monomials)]
        degrees = [degree_distribution() for _ in range(n_monomials)]
        monomials = [Monomial.from_random_process(max_index, degree, sigma)
                     for degree in degrees]
        
        return cls(monomials, coefficients)
    
    def degrees(self):
        return [monomial.degree() for monomial in self.monomials]
    
    def __len__(self):
        return len(self.monomials)
    
    def __str__(self):
        factors = []
        for coefficient, monomial in zip(self.coefficients, self.monomials):
            factors.append(f"{coefficient:}" + (f"*{monomial}" if monomial.degree() else ""))
        return " + ".join(factors)
    
    def __repr__(self):
        return str(self)
    
    def latex_string(self, precision=3):
        factors = []
        for coefficient, monomial in zip(self.coefficients, self.monomials):
            if coefficient == 0:
                continue
            if factors:
                factors.append(' + ' if coefficient > 0 else ' - ')
            elif coefficient < 0:
                factors.append(' -')
            factors.append(f"{abs(coefficient):.{precision}f}{monomial.latex_string()}")
        return ''.join(factors)
    
    def get_exponents(self, n_exponents=None):
        if not n_exponents:
            n_exponents = 1 + max(max(m.indexes) for m in self.monomials)
        exponents = np.stack([m.get_exponents(n_exponents) for m in self.monomials])
        return exponents
