classdef PropAnalysisLinearity < matlab.unittest.TestCase
%PROPANALYSISLINEARITY  Linearity of the filterbank analysis operator.
%
%   Property: filterbank(α·x + β·y, g, a) ≡ α·filterbank(x,g,a) + β·filterbank(y,g,a)
%
%   The filterbank analysis T is a linear map.  This is tested by verifying
%   superposition for random pairs (x, y) and random scalars (α, β), using
%   both real and complex inputs.

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

        function testLinearityRealInputs(tc)
            % α·x + β·y with complex scalars, real signals
            [g, a, ~, L] = audfilters(tc.p.fs, tc.p.Ls); %#ok<ASGLU>
            Ls = tc.p.Ls;

            for trial = 1:50
                x     = randn(Ls, 1);
                y     = randn(Ls, 1);
                alpha = randn + 1i*randn;
                beta  = randn + 1i*randn;

                cx    = filterbank(x, g, a);
                cy    = filterbank(y, g, a);
                c_sum = filterbank(alpha*x + beta*y, g, a);

                M      = numel(cx);
                maxErr = 0;
                for m = 1:M
                    denom = norm(cx{m}) + norm(cy{m}) + eps;
                    err   = norm(c_sum{m} - (alpha*cx{m} + beta*cy{m})) / denom;
                    maxErr = max(maxErr, err);
                end

                tc.verifyLessThan(maxErr, 1e-10, ...
                    sprintf('Trial %d (real inputs): linearity error %.2e', trial, maxErr));
            end
        end

        function testLinearityComplexInputs(tc)
            % α·x + β·y with complex scalars and complex signals
            [g, a, ~, L] = audfilters(tc.p.fs, tc.p.Ls); %#ok<ASGLU>
            Ls = tc.p.Ls;

            for trial = 1:50
                x     = randn(Ls,1) + 1i*randn(Ls,1);
                y     = randn(Ls,1) + 1i*randn(Ls,1);
                alpha = randn + 1i*randn;
                beta  = randn + 1i*randn;

                cx    = filterbank(x, g, a);
                cy    = filterbank(y, g, a);
                c_sum = filterbank(alpha*x + beta*y, g, a);

                M      = numel(cx);
                maxErr = 0;
                for m = 1:M
                    denom = norm(cx{m}) + norm(cy{m}) + eps;
                    err   = norm(c_sum{m} - (alpha*cx{m} + beta*cy{m})) / denom;
                    maxErr = max(maxErr, err);
                end

                tc.verifyLessThan(maxErr, 1e-10, ...
                    sprintf('Trial %d (complex inputs): linearity error %.2e', trial, maxErr));
            end
        end

        function testScalarMultiple(tc)
            % Scaling the input by s must scale every subband coefficient by s.
            [g, a, ~, L] = audfilters(tc.p.fs, tc.p.Ls); %#ok<ASGLU>
            Ls = tc.p.Ls;

            scales = [2, -1, 1i, -3+2i, 0.5-0.5i];
            for k = 1:numel(scales)
                s = scales(k);
                x   = randn(Ls,1) + 1i*randn(Ls,1);
                cx  = filterbank(x,   g, a);
                csx = filterbank(s*x, g, a);
                M = numel(cx);
                for m = 1:M
                    err = norm(csx{m} - s*cx{m}) / (norm(cx{m}) + eps);
                    tc.verifyLessThan(err, 1e-12, ...
                        sprintf('Scale s=%g+%gi, band %d: relative error %.2e', ...
                        real(s), imag(s), m, err));
                end
            end
        end

        function testZeroInputGivesZeroCoefficients(tc)
            % filterbank(0, g, a) must produce all-zero coefficient cells.
            [g, a, ~, L] = audfilters(tc.p.fs, tc.p.Ls); %#ok<ASGLU>
            Ls = tc.p.Ls;
            x0 = zeros(Ls, 1);
            c0 = filterbank(x0, g, a);
            M  = numel(c0);
            for m = 1:M
                tc.verifyLessThan(norm(c0{m}), 1e-12, ...
                    sprintf('Band %d: filterbank(0) should be zero', m));
            end
        end

    end
end
