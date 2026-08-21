classdef PropFrameOperatorSymmetry < matlab.unittest.TestCase
%PROPFRAMEOPERATORSYMMETRY  Self-adjointness and PSD of the frame operator.
%
%   The frame operator S_g is defined as the composition of the synthesis
%   operator (ifilterbank) applied to the analysis output (filterbank):
%
%     S_g x  =  ifilterbank( filterbank(x, g, a), g, a, L )
%
%   Since S_g = T_g* T_g (analysis adjoint composed with analysis), it is:
%
%   (1) Self-adjoint:         <S_g x, y>  =  <x, S_g y>
%   (2) Positive semi-definite: <S_g x, x> >= 0
%   (3) S_g applied to the zero signal returns the zero signal.

    properties
        p   % scalar parameters (fs, Ls)
    end

    methods (TestClassSetup)
        function setupClass(tc)
            sharedDir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'shared');
            addpath(sharedDir);
            setup_filterbank_paths();
            rng(42);
            tc.p = struct('fs', 8000, 'Ls', 1024);
        end
    end

    methods (Test)

        function testSelfAdjoint(tc)
            % <S_g x, y> == <x, S_g y>  for random complex x, y.
            [g, a, ~, L] = audfilters(tc.p.fs, tc.p.Ls);
            Ls = tc.p.Ls;

            for trial = 1:100
                x   = randn(Ls,1) + 1i*randn(Ls,1);
                y   = randn(Ls,1) + 1i*randn(Ls,1);

                Sgx = ifilterbank(filterbank(x, g, a), g, a, L);
                Sgy = ifilterbank(filterbank(y, g, a), g, a, L);

                % Trim to Ls in case ifilterbank returns length L > Ls
                Sgx = Sgx(1:Ls);
                Sgy = Sgy(1:Ls);

                lhs = dot(Sgx, y);   % <S_g x, y>
                rhs = dot(x, Sgy);   % <x, S_g y>

                err = abs(lhs - rhs) / (abs(lhs) + abs(rhs) + eps);
                tc.verifyLessThan(err, 1e-8, ...
                    sprintf('Trial %d: self-adjoint error %.2e (lhs=%.4e, rhs=%.4e)', ...
                    trial, err, abs(lhs), abs(rhs)));
            end
        end

        function testPositiveSemiDefinite(tc)
            % <S_g x, x> >= 0  (S_g = T*T is always PSD).
            [g, a, ~, L] = audfilters(tc.p.fs, tc.p.Ls);
            Ls = tc.p.Ls;

            for trial = 1:50
                x   = randn(Ls,1) + 1i*randn(Ls,1);
                Sgx = ifilterbank(filterbank(x, g, a), g, a, L);
                Sgx = Sgx(1:Ls);

                innerProd = real(dot(Sgx, x));   % must be real and ≥ 0
                tc.verifyGreaterThanOrEqual(innerProd, -1e-10, ...
                    sprintf('Trial %d: <S_g x, x> = %.4e < 0 (not PSD)', trial, innerProd));
            end
        end

        function testZeroSignalGivesZero(tc)
            % S_g applied to the zero signal must return zero.
            [g, a, ~, L] = audfilters(tc.p.fs, tc.p.Ls);
            Ls = tc.p.Ls;

            x0  = zeros(Ls, 1);
            cx0 = filterbank(x0, g, a);
            Sg0 = ifilterbank(cx0, g, a, L);

            tc.verifyLessThan(norm(Sg0), 1e-12, ...
                'S_g applied to zero signal must give zero output');
        end

        function testFrameOperatorIsLinear(tc)
            % S_g is linear: S_g(α x + β y) = α S_g x + β S_g y
            [g, a, ~, L] = audfilters(tc.p.fs, tc.p.Ls);
            Ls = tc.p.Ls;

            for trial = 1:30
                x     = randn(Ls,1) + 1i*randn(Ls,1);
                y     = randn(Ls,1) + 1i*randn(Ls,1);
                alpha = randn + 1i*randn;
                beta  = randn + 1i*randn;

                Sgx = ifilterbank(filterbank(x, g, a), g, a, L);
                Sgy = ifilterbank(filterbank(y, g, a), g, a, L);
                Sgz = ifilterbank(filterbank(alpha*x + beta*y, g, a), g, a, L);

                Sgx = Sgx(1:Ls);
                Sgy = Sgy(1:Ls);
                Sgz = Sgz(1:Ls);

                expected = alpha*Sgx + beta*Sgy;
                err = norm(Sgz - expected) / (norm(expected) + eps);
                tc.verifyLessThan(err, 1e-10, ...
                    sprintf('Trial %d: frame operator linearity error %.2e', trial, err));
            end
        end

    end
end
