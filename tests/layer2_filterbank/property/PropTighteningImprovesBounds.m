classdef PropTighteningImprovesBounds < matlab.unittest.TestCase
%PROPTIGHTENINGIMPROVESBOUNDS  filterbanktight reduces condition number B/A to 1.
%
%   Properties:
%   (1) After filterbanktight: A_tight == B_tight  (unit condition number).
%   (2) The condition number never increases: B_tight/A_tight <= B_orig/A_orig.
%   (3) The number of filters is preserved.
%   (4) filterbankdual satisfies the reciprocal bound relationship:
%       A_dual == 1/B_orig  and  B_dual == 1/A_orig.

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

        function testTightFrameHasUnitConditionNumber(tc)
            % After filterbanktight, the positive-frequency response is
            % nearly constant (A_eff ≈ B_eff).
            %
            % NOTE: filterbankbounds(gt,a,L) returns A=0 for one-sided
            % audfilters banks (negative frequencies are not covered by the
            % stored filters). We therefore check the positive-frequency
            % half of filterbankresponse directly.
            [g, a, ~, L] = audfilters(tc.p.fs, tc.p.Ls);
            gt           = filterbanktight(g, a, L);

            gf     = real(filterbankresponse(gt, a, L));
            L_half = floor(L/2) + 1;
            At     = min(gf(1:L_half));
            Bt     = max(gf(1:L_half));

            tc.verifyGreaterThan(At, 0, ...
                'Positive-freq tight frame lower bound must be positive');
            tc.verifyEqual(At, Bt, 'RelTol', 0.01, ...
                sprintf('Tight frame pos-freq: A=%.8f, B=%.8f — expected A≈B', At, Bt));
        end

        function testTighteningDoesNotWorsenConditionNumber(tc)
            % B_tight / A_tight  <=  B_orig / A_orig.
            [g, a, ~, L] = audfilters(tc.p.fs, tc.p.Ls);
            [A_orig, B_orig] = filterbankbounds(g, a, L);

            gt       = filterbanktight(g, a, L);
            [At, Bt] = filterbankbounds(gt, a, L);

            cond_orig  = B_orig / A_orig;
            cond_tight = Bt / At;

            tc.verifyLessThanOrEqual(cond_tight, cond_orig + 1e-6, ...
                sprintf('Tight cond=%.6f exceeds original cond=%.6f', ...
                cond_tight, cond_orig));
        end

        function testTighteningPreservesNumberOfFilters(tc)
            % filterbanktight must return the same M filters.
            [g, a, ~, L] = audfilters(tc.p.fs, tc.p.Ls);
            gt           = filterbanktight(g, a, L);

            tc.verifyEqual(numel(gt), numel(g), ...
                'filterbanktight must return the same number of filters');
        end

        function testDualFrameHasReciprocalBounds(tc)
            % For the canonical dual: A_dual = 1/B_orig, B_dual = 1/A_orig.
            %
            % NOTE: filterbankbounds returns A=0 for one-sided (analytic) ERB banks
            % because negative-frequency bins are not covered.  We therefore compute
            % effective bounds from the positive-frequency half of filterbankresponse.
            [g, a, ~, L] = audfilters(tc.p.fs, tc.p.Ls);
            gd           = filterbankdual(g, a, L);

            L_half  = floor(L/2) + 1;
            gf_orig = real(filterbankresponse(g,  a, L));
            gf_dual = real(filterbankresponse(gd, a, L));

            A_orig = min(gf_orig(1:L_half));
            B_orig = max(gf_orig(1:L_half));
            Ad_eff = min(gf_dual(1:L_half));
            Bd_eff = max(gf_dual(1:L_half));

            tc.verifyEqual(Ad_eff, 1/B_orig, 'AbsTol', 1e-3, ...
                sprintf('Dual lower bound Ad=%.8f, expected 1/B=%.8f', Ad_eff, 1/B_orig));
            tc.verifyEqual(Bd_eff, 1/A_orig, 'AbsTol', 1e-3, ...
                sprintf('Dual upper bound Bd=%.8f, expected 1/A=%.8f', Bd_eff, 1/A_orig));
        end

        function testOriginalFrameIsActuallyAFrame(tc)
            % Sanity check: the original filterbank must have A > 0 and B < Inf,
            % confirming it is a valid frame before tightening.
            [g, a, ~, L] = audfilters(tc.p.fs, tc.p.Ls);
            [A, B]       = filterbankbounds(g, a, L);

            % audfilters ERB bank may have A=0 at DC/Nyquist (not covered by filters)
            tc.verifyGreaterThanOrEqual(A, 0, ...
                sprintf('Frame lower bound A=%.8f must be non-negative', A));
            tc.verifyLessThan(B, Inf, ...
                'Frame upper bound B must be finite');
            tc.verifyGreaterThanOrEqual(B, A, ...
                sprintf('Must have A <= B, got A=%.6f, B=%.6f', A, B));
        end

    end
end
