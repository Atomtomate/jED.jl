abstract type Model end

struct AIM <: Model
    NSites::Int
    tMatrix::SMatrix{Float64}
    UMatrix::SMatrix{Float64}

    function AIM(NSites)
    end
end
