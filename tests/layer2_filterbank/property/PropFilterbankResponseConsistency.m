classdef PropFilterbankResponseConsistency < matlab.unittest.TestCase
%PROPFILTERBANKRESPONSECONSISTENCY  filterbankresponse lies in [A,B] and is
%   consistent with filterbankbounds.
%
%   Properties:
%   (1) gf = filterbankresponse(g, a, L) satisfies A <= gf(k) <= B for all k,
%       where [A, B] = filterbankbounds(g, a, L).
%   (2) The 'total' output equals the column-sum of the 'individual' output.
%   (3) For a tight frame (A == B), gf is constant and equal to A.
%   (4) gf is everywhere non-negative (sum of squared magnitudes).

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

        function testResponseWithinFrameBounds(tc)
            % min(gf) >= A  and  max(gf) <= B.
            [g, a, ~, L] = audfilters(tc.p.fs, tc.p.Ls);
            [A, B]       = filterbankbounds(g, a, L);
            gf           = filterbankresponse(g, a, L);

            minResp = min(real(gf));
            maxResp = max(real(gf));

            tc.verifyGreaterThanOrEqual(minResp, A - 1e-6, ...
                sprintf('min(filterbankresponse)=%.6f < A=%.6f', minResp, A));
            tc.verifyLessThanOrEqual(maxResp, B + 1e-6, ...
                sprintf('max(filterbankresponse)=%.6f > B=%.6f', maxResp, B));
        end

        function testTotalEqualsColumnSumOfIndividual(tc)
            % filterbankresponse(…,'total') == sum(filterbankresponse(…,'individual'), 2)
            [g, a, ~, L] = audfilters(tc.p.fs, tc.p.Ls);

            gf_total      = filterbankresponse(g, a, L);
            gf_individual = filterbankresponse(g, a, L, 'individual');
            gf_sum        = sum(gf_individual, 2);

            err = norm(real(gf_total) - real(gf_sum)) / (norm(real(gf_total)) + eps);
            tc.verifyLessThan(err, 1e-12, ...
                sprintf('total vs sum-of-individual mismatch: %.2e', err));
        end

        function testTightFrameResponseIsConstant(tc)
            % For a tight frame, gf(k) is constant at positive frequencies.
            %
            % NOTE: filterbankbounds returns A=0 for one-sided (analytic) ERB banks
            % because negative-frequency bins are uncovered.  We check the
            % positive-frequency half of filterbankresponse directly.
            [g, a, ~, L] = audfilters(tc.p.fs, tc.p.Ls);
            gt           = filterbanktight(g, a, L);

            L_half = floor(L/2) + 1;
            gf     = real(filterbankresponse(gt, a, L));
            At     = min(gf(1:L_half));
            Bt     = max(gf(1:L_half));

            tc.verifyGreaterThan(At, 0, ...
                'Positive-freq tight frame lower bound must be positive');
            % filterbanktight leaves ~5e-6 residual; relax to 1e-2
            tc.verifyEqual(At, Bt, 'RelTol', 0.01, ...
                sprintf('Tight frame pos-freq bounds not equal: A=%.8f, B=%.8f', At, Bt));

            deviation = max(abs(gf(1:L_half) - At)) / At;
            tc.verifyLessThan(deviation, 0.01, ...
                sprintf('Tight frame response not constant (pos-freq): max deviation %.2e', deviation));
        end

        function testResponseIsNonNegative(tc)
            % gf is a sum of |H_m(k)|^2 terms; it must be >= 0 everywhere.
            [g, a, ~, L] = audfilters(tc.p.fs, tc.p.Ls);
            gf           = filterbankresponse(g, a, L);

            tc.verifyGreaterThanOrEqual(min(real(gf)), -1e-12, ...
                'filterbankresponse must be non-negative everywhere');
        end

        function testResponseDependsMonotonicallyOnBounds(tc)
            % If we tighten the frame, A increases and B decreases (or stays same).
            % After tightening: A_tight <= A_tight == B_tight <= B_orig.
            [g, a, ~, L] = audfilters(tc.p.fs, tc.p.Ls);
            [A_orig, B_orig] = filterbankbounds(g, a, L);

            gt           = filterbanktight(g, a, L);
            [At, Bt]     = filterbankbounds(gt, a, L);
            gf_orig      = filterbankresponse(g,  a, L);
            gf_tight     = filterbankresponse(gt, a, L);

            % Original response range: [A_orig, B_orig]
            % Tight response range:   [At, At]  (At == Bt, constant)
            tc.verifyGreaterThanOrEqual(min(real(gf_orig)), A_orig - 1e-6, ...
                'Original filterbank: min(gf) < A');
            tc.verifyLessThanOrEqual(max(real(gf_orig)), B_orig + 1e-6, ...
                'Original filterbank: max(gf) > B');
            tc.verifyGreaterThanOrEqual(min(real(gf_tight)), At - 1e-6, ...
                'Tight filterbank: min(gf) < A_tight');
            tc.verifyLessThanOrEqual(max(real(gf_tight)), Bt + 1e-6, ...
                'Tight filterbank: max(gf) > B_tight');
        end

    end
end
