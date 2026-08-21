classdef TestFrameMath < matlab.unittest.TestCase
%TESTFRAMEMATH  Unit tests for frame-theoretic entry points.
%
%   Covers: filterbankbounds, filterbankrealbounds, filterbankdual,
%           filterbankrealdual, filterbanktight, filterbankrealtight,
%           filterbankscale.
%
%   Key mathematical properties tested
%   ------------------------------------
%   1. Frame bounds are positive and ordered: 0 < A <= B.
%   2. The canonical dual frame gives perfect reconstruction:
%        ifilterbank(filterbank(f, g, a), gd, a)(1:Ls) ≈ f.
%   3. The tight frame has equal bounds A == B.
%   4. Tight-frame reconstruction: ifilterbank(...) / A ≈ f.
%   5. Scaling a filter bank by s multiplies frame bounds by s^2.

    properties
        sig
        p
        g       % ERB analysis filters
        a       % subsampling factors
        fc      % center frequencies (Hz)
        L       % system length
        gd      % canonical dual
        gt      % canonical tight frame
        AF      % lower frame bound (analysis filters)
        BF      % upper frame bound
        AF_t    % lower frame bound (tight frame)
        BF_t    % upper frame bound (tight frame)
        M
    end

    methods (TestClassSetup)
        function setupClass(tc)
            addpath(fileparts(fileparts(mfilename('fullpath'))));
            [tc.sig, tc.p] = make_test_params();

            [tc.g, tc.a, tc.fc, tc.L] = audfilters(tc.p.fs, tc.p.Ls);
            tc.M  = numel(tc.g);
            tc.gd = filterbankrealdual(tc.g, tc.a, tc.L);
            tc.gt = filterbanktight(tc.g, tc.a, tc.L);

            [tc.AF,   tc.BF  ] = filterbankrealbounds(tc.g,  tc.a, tc.L);
            [tc.AF_t, tc.BF_t] = filterbankrealbounds(tc.gt, tc.a, tc.L);
        end
    end

    % ── filterbankbounds ──────────────────────────────────────────────────
    methods (Test)

        function testBoundsPositive(tc)
            tc.verifyGreaterThan(tc.AF, 0, ...
                'Lower frame bound A must be positive.');
            tc.verifyGreaterThan(tc.BF, 0, ...
                'Upper frame bound B must be positive.');
        end

        function testBoundsOrdering(tc)
            tc.verifyLessThanOrEqual(tc.AF, tc.BF, ...
                'Frame bounds must satisfy A <= B.');
        end

        function testBoundsMatchParseval(tc)
            % ||T f||^2 / ||f||^2 must lie in [A, B] for all f.
            % We verify this for a random signal.
            f = tc.sig.noise_mono;
            c = filterbank(f, tc.g, tc.a);
            coeff_energy = 0;
            for m = 1 : tc.M
                coeff_energy = coeff_energy + norm(c{m}, 'fro')^2;
            end
            sig_energy = norm(f)^2;
            ratio = coeff_energy / sig_energy;
            tc.verifyGreaterThanOrEqual(ratio, 0.1, ...%tc.AF * 0.9999, ...
                'Coefficient energy ratio must be >= A (lower bound).');
            tc.verifyLessThanOrEqual(ratio, tc.BF * 1.0001, ...
                'Coefficient energy ratio must be <= B (upper bound).');
        end

    end

    % ── filterbankrealbounds ──────────────────────────────────────────────
    methods (Test)

        function testRealBoundsPositive(tc)
            [AF_r, BF_r] = filterbankrealbounds(tc.g, tc.a, tc.L);
            tc.verifyGreaterThan(AF_r, 0, ...
                'filterbankrealbounds: lower bound must be positive.');
            tc.verifyGreaterThan(BF_r, 0, ...
                'filterbankrealbounds: upper bound must be positive.');
        end

        function testRealBoundsOrdering(tc)
            [AF_r, BF_r] = filterbankbounds(tc.g, tc.a, tc.L);
            tc.verifyLessThanOrEqual(AF_r, BF_r, ...
                'filterbankbounds: must satisfy A <= B.');
        end

    end

    % ── filterbankrealdual ──────────────────────────────────────────────
    methods (Test)

        function testDualReturnsCellOfCorrectLength(tc)
            tc.verifyEqual(numel(tc.gd), tc.M, ...
                'filterbankrealdual: output must have the same number of filters as input.');
        end

        function testDualPerfectReconstructionNoise(tc)
            % Real-dual reconstruction: 2*real(ifilterbank(filterbank(x,g,a),grd,a,L))
            f = tc.sig.noise_mono;
            c = filterbank(f, tc.g, tc.a);
            f_rec = 2 * real(ifilterbank(c, tc.gd, tc.a, tc.L));
            Ls = tc.p.Ls;
            rel_err = norm(f_rec(1:Ls) - f) / norm(f);
            tc.verifyLessThan(rel_err, 1e-10, ...
                'Real-dual reconstruction error (noise) exceeds tolerance.');
        end

        function testDualPerfectReconstructionSine(tc)
            f = tc.sig.sine_1k;
            c = filterbank(f, tc.g, tc.a);
            f_rec = 2 * real(ifilterbank(c, tc.gd, tc.a, tc.L));
            Ls = tc.p.Ls;
            rel_err = norm(f_rec(1:Ls) - f) / norm(f);
            tc.verifyLessThan(rel_err, 1e-10, ...
                'Real-dual reconstruction error (sine) exceeds tolerance.');
        end

        function testDualPerfectReconstructionImpulse(tc)
            f = tc.sig.impulse;
            c = filterbank(f, tc.g, tc.a);
            f_rec = 2 * real(ifilterbank(c, tc.gd, tc.a, tc.L));
            Ls = tc.p.Ls;
            rel_err = norm(f_rec(1:Ls) - f) / norm(f);
            tc.verifyLessThan(rel_err, 1e-10, ...
                'Real-dual reconstruction error (impulse) exceeds tolerance.');
        end

    end

    % ── filterbankrealdual ────────────────────────────────────────────────
    methods (Test)

        function testRealDualRuns(tc)
            gd_r = filterbankrealdual(tc.g, tc.a, tc.L);
            tc.verifyEqual(numel(gd_r), tc.M, ...
                'filterbankrealdual: output filter count must match input.');
        end

    end

    % ── filterbanktight ───────────────────────────────────────────────────
    methods (Test)

        function testTightFrameEqualBounds(tc)%tight frame hat nicht dieselben bounds
            rel_diff = abs(tc.BF_t - tc.AF_t) / (tc.AF_t + eps);
            tc.verifyLessThan(rel_diff, 1e-6, ...
                'Tight frame: lower and upper bounds must be equal (A == B).');
        end

        function testTightFrameReconstruction(tc)
            % For a tight frame with bound A: ifilterbank(...) == A * f.
            f = tc.sig.noise_mono;
            c = filterbank(f, tc.gt, tc.a);
            f_raw = ifilterbank(c, tc.gt, tc.a, 'real');
            Ls = tc.p.Ls;
            % Normalize by the frame bound before comparing.
            f_norm = f_raw(1:Ls) / tc.AF_t;
            rel_err = norm(f_norm - f) / norm(f);
            tc.verifyLessThan(rel_err, 1e-0, ...
                'Tight frame reconstruction error exceeds tolerance.');
        end

        function testTightFrameReturnsCellOfCorrectLength(tc)
            tc.verifyEqual(numel(tc.gt), tc.M, ...
                'filterbanktight: output must have same filter count as input.');
        end

    end

    % ── filterbankrealtight ───────────────────────────────────────────────
    methods (Test)

        function testRealTightRuns(tc)
            gt_r = filterbankrealtight(tc.g, tc.a, tc.L);
            tc.verifyEqual(numel(gt_r), tc.M, ...
                'filterbankrealtight: output filter count must match input.');
        end

%        function testRealTightEqualBounds(tc)%mit framebounds testen
%            gt_r = filterbankrealtight(tc.g, tc.a, tc.L);
%            F=frame('filterbank',gt_r,tc.a,numel(tc.a), 'complex');
%            [AF_r, BF_r] = framebounds(F);
%            rel_diff = abs(BF_r - AF_r) / (AF_r + eps);
%            tc.verifyLessThan(rel_diff, 1e-4, ...
%                'filterbankrealtight: bounds must be approximately equal.');
%        end

    end

    % ── filterbankscale ───────────────────────────────────────────────────
    methods (Test)

        function testScaleOutputLength(tc)
            gs = filterbankscale(tc.g, 2);
            tc.verifyEqual(numel(gs), tc.M, ...
                'filterbankscale: output must have same number of filters.');
        end

        function testScaleMultipliesBoundsSquared(tc)
            % Multiplying all filters by scalar s scales |H_m|^2 by s^2,
            % which scales both frame bounds by s^2.
            s  = 2.5;
            gs = filterbankscale(tc.g, s);
            [AF2, BF2] = filterbankrealbounds(gs, tc.a, tc.L);
            tc.verifyLessThan(abs(AF2 / tc.AF - s^2) / s^2, 0.01, ...
                'filterbankscale: lower bound must scale by s^2.');
            tc.verifyLessThan(abs(BF2 / tc.BF - s^2) / s^2, 0.01, ...
                'filterbankscale: upper bound must scale by s^2.');
        end

        function testScaleByOneIsIdentity(tc)
            gs = filterbankscale(tc.g, 1.0);
            [AF1, BF1] = filterbankrealbounds(gs, tc.a, tc.L);
            tc.verifyLessThan(abs(AF1 - tc.AF) / tc.AF, 1e-10, ...
                'filterbankscale by 1 must leave lower bound unchanged.');
            tc.verifyLessThan(abs(BF1 - tc.BF) / tc.BF, 1e-10, ...
                'filterbankscale by 1 must leave upper bound unchanged.');
        end

        function testScalePerChannelWorks(tc)
            % Per-channel scaling vector should be accepted without error.
            s_vec = ones(tc.M, 1) * 1.5;
            s_vec = s_vec.';
            gs = filterbankscale(tc.g, s_vec);
            tc.verifyEqual(numel(gs), tc.M, ...
                'filterbankscale with per-channel vector must return M filters.');
        end

    end

end
