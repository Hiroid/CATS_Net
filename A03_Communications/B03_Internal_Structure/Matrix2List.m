function List = Matrix2List (Matrix)
    [a,b] = size(Matrix);
    List = Matrix(tril(ones(a,b),-1)>0); 
end