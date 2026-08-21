classdef TestRtpghifb < matlab.unittest.TestCase
%TESTRTPGHIFB  Unit tests for Real-Time Phase Gradient Heap Integration
%              for filter banks (RTPGHIFB).
%
%   Covers: rtpghifbwl
%
%   The filterbank is built with waveletfilters (constant-Q scales), which
%   produces the uniform hop-size and normalised centre-frequency vector
%   (info.fc) and time-frequency ratio handle (info.tfr) required by
%   rtpghifbwl.
%
%   Input convention (matches the existing test_rtpghifb.m reference):
%       corig = ufilterbank(f, g, a)        % M x N complex coefficients
%       s     = abs(corig.')                % N x M  -> transposed to M x N
%       [c, newphase, tgrad, fgrad] = rtpghifbwl(s, a(1), info.fc, info.tfr)

    properties
        sig     % test signal struct  (from make_test_params)
        p       % parameter struct (fs, Ls, tol, abs_tol)
        g       % wavelet analysis filters
        a       % hop sizes (uniform scalar, but stored as vector)
        fc      % normalised centre frequencies (info.fc)
        tfr     % time-frequency ratio handle  (info.tfr)
        L       % system length
        M       % number of channels
        corig   % reference coefficients  (ufilterbank output)
        s       % magnitude input to rtpghifbwl (M x N double)
        f       % single test signal (column, length Ls)
    end

    methods (TestClassSetup)
        function setupClass(tc)
            sharedDir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'shared');
            addpath(sharedDir);
            setup_filterbank_paths();
            [tc.sig, tc.p] = make_test_params();

            % Build a moderate-size constant-Q wavelet filterbank.
            Ls     = tc.p.Ls;
            scales = 2.^(linspace(5, -2, 64));   % 64 channels, fast enough for unit tests
            [tc.g, tc.a, ~, tc.L, info] = ...
                waveletfilters(Ls, scales, 'repeat', 'uniform');

            tc.M   = numel(tc.g);
            tc.fc  = info.fc;
            tc.tfr = info.tfr;

            tc.f     = tc.sig.noise_mono;
            tc.corig = ufilterbank(tc.f, tc.g, tc.a);
            % rtpghifbwl expects magnitude in (M x N) layout.
            tc.s     = abs(tc.corig.');
        end
    end

    % ── Output structure ──────────────────────────────────────────────────
    methods (Test)

        function testFourOutputsReturnedWithoutError(tc)
            % The function must return four outputs without error or warning.
            tc.verifyWarningFree( ...
                @() rtpghifbwl(tc.s, tc.a(1), tc.fc, tc.tfr), ...
                'rtpghifbwl: should run without warnings.');
            [c, newphase, tgrad, fgrad] = rtpghifbwl(tc.s, tc.a(1), tc.fc, tc.tfr);
            tc.verifyNotEmpty(c,        'rtpghifbwl: c must not be empty.');
            tc.verifyNotEmpty(newphase, 'rtpghifbwl: newphase must not be empty.');
            tc.verifyNotEmpty(tgrad,    'rtpghifbwl: tgrad must not be empty.');
            tc.verifyNotEmpty(fgrad,    'rtpghifbwl: fgrad must not be empty.');
        end

        function testOutputSizeMatchesInput(tc)
            % All four outputs must have the same size as s (M x N).
            [c, newphase, tgrad, fgrad] = rtpghifbwl(tc.s, tc.a(1), tc.fc, tc.tfr);
            expectedSize = size(tc.s);
            tc.verifyEqual(size(c),        expectedSize, ...
                'rtpghifbwl: c must have the same size as input s.');
            tc.verifyEqual(size(newphase), expectedSize, ...
                'rtpghifbwl: newphase must have the same size as input s.');
            tc.verifyEqual(size(tgrad),    expectedSize, ...
                'rtpghifbwl: tgrad must have the same size as input s.');
            tc.verifyEqual(size(fgrad),    expectedSize, ...
                'rtpghifbwl: fgrad must have the same size as input s.');
        end

        function testOutputIsComplex(tc)
            % c must be complex (it contains reconstructed coefficients).
            [c, ~, ~, ~] = rtpghifbwl(tc.s, tc.a(1), tc.fc, tc.tfr);
            tc.verifyFalse(isreal(c), ...
                'rtpghifbwl: c must be complex.');
        end

        function testPhaseIsReal(tc)
            % newphase must be real (it is an angle in radians).
            [~, newphase, ~, ~] = rtpghifbwl(tc.s, tc.a(1), tc.fc, tc.tfr);
            tc.verifyTrue(isreal(newphase), ...
                'rtpghifbwl: newphase must be real.');
        end

        function testGradientsAreReal(tc)
            % tgrad and fgrad are phase derivatives and must be real.
            [~, ~, tgrad, fgrad] = rtpghifbwl(tc.s, tc.a(1), tc.fc, tc.tfr);
            tc.verifyTrue(isreal(tgrad), ...
                'rtpghifbwl: tgrad must be real.');
            tc.verifyTrue(isreal(fgrad), ...
                'rtpghifbwl: fgrad must be real.');
        end

    end

    % ── Magnitude preservation ────────────────────────────────────────────
    methods (Test)

        function testMagnitudeOfCMatchesInput(tc)
            % The fundamental PGHI property: |c(m,n)| = s(m,n).
            [c, ~, ~, ~] = rtpghifbwl(tc.s, tc.a(1), tc.fc, tc.tfr);
            magErr = norm(abs(c(:)) - tc.s(:)) / (norm(tc.s(:)) + eps);
            tc.verifyLessThan(magErr, 1e-6, ...
                'rtpghifbwl: |c| must equal the input magnitude s.');
        end

        function testMagnitudeErrDbIsSmall(tc)
            % magnitudeerrdb should be well below 0 dB (i.e. not garbage).
            % This mirrors the reference test_rtpghifb.m sanity check.
            [c, ~, ~, ~] = rtpghifbwl(tc.s, tc.a(1), tc.fc, tc.tfr);
            % Reconstruct with ifilterbank to get audio, then re-analyse.
            gd      = filterbankrealdual(tc.g, tc.a, tc.L);
            f_rec   = ifilterbank(c.', gd, tc.a, 'real');
            c_rec   = ufilterbank(f_rec, tc.g, tc.a);
            errdb   = magnitudeerrdb(tc.corig, c_rec);
            % At minimum the error should be finite and < 0 dB.
            tc.verifyTrue(isfinite(errdb), ...
                'rtpghifbwl: magnitude reconstruction error must be finite.');
            tc.verifyLessThan(errdb, 0, ...
                'rtpghifbwl: magnitude reconstruction error must be < 0 dB.');
        end

        function testCEqualsMagnitudeTimesExpPhase(tc)
            % c must factor exactly as s .* exp(1i * newphase).
            [c, newphase, ~, ~] = rtpghifbwl(tc.s, tc.a(1), tc.fc, tc.tfr);
            c_expected = tc.s .* exp(1i * newphase);
            relErr = norm(c(:) - c_expected(:)) / (norm(c_expected(:)) + eps);
            tc.verifyLessThan(relErr, 1e-10, ...
                'rtpghifbwl: c must equal s .* exp(1i * newphase).');
        end

    end

    % ── Normal vs causal variant ──────────────────────────────────────────
    methods (Test)

        function testCausalVariantRunsWithoutError(tc)
            % The 'causal' flag must be accepted and produce output.
            tc.verifyWarningFree( ...
                @() rtpghifbwl(tc.s, tc.a(1), tc.fc, tc.tfr, 'causal'), ...
                'rtpghifbwl: causal variant should run without warnings.');
        end

        function testNormalVariantRunsWithoutError(tc)
            % The default ('normal') flag must be accepted.
            tc.verifyWarningFree( ...
                @() rtpghifbwl(tc.s, tc.a(1), tc.fc, tc.tfr, 'normal'), ...
                'rtpghifbwl: normal variant should run without warnings.');
        end

        function testCausalPreservesMagnitude(tc)
            % The causal variant must also satisfy |c| == s.
            [c_caus, ~, ~, ~] = rtpghifbwl(tc.s, tc.a(1), tc.fc, tc.tfr, 'causal');
            magErr = norm(abs(c_caus(:)) - tc.s(:)) / (norm(tc.s(:)) + eps);
            tc.verifyLessThan(magErr, 1e-6, ...
                'rtpghifbwl (causal): |c| must equal the input magnitude s.');
        end

        function testCausalAndNormalOutputsDiffer(tc)
            % Causal and normal variants use different phase integration
            % schemes and should generally produce different phases.
            [c_norm, ~, ~, ~] = rtpghifbwl(tc.s, tc.a(1), tc.fc, tc.tfr, 'normal');
            [c_caus, ~, ~, ~] = rtpghifbwl(tc.s, tc.a(1), tc.fc, tc.tfr, 'causal');
            phaseDiff = angle(c_norm(:)) - angle(c_caus(:));
            % Not all phases need to differ, but the total variation should
            % be non-trivial (not identically zero).
            tc.verifyGreaterThan(norm(phaseDiff), 1e-6, ...
                'rtpghifbwl: normal and causal variants must produce different phases.');
        end

    end

    % ── Tolerance parameter ───────────────────────────────────────────────
    methods (Test)

        function testCustomTolAccepted(tc)
            % Passing a custom tolerance must not error.
            tc.verifyWarningFree( ...
                @() rtpghifbwl(tc.s, tc.a(1), tc.fc, tc.tfr, 'tol', 1e-3), ...
                'rtpghifbwl: custom tol should be accepted.');
        end

        function testHighTolPreservesMagnitude(tc)
            % Even with a loose tolerance the magnitude constraint must hold.
            [c_ht, ~, ~, ~] = rtpghifbwl(tc.s, tc.a(1), tc.fc, tc.tfr, 'tol', 1e-3);
            magErr = norm(abs(c_ht(:)) - tc.s(:)) / (norm(tc.s(:)) + eps);
            tc.verifyLessThan(magErr, 1e-6, ...
                'rtpghifbwl (high tol): |c| must still equal input magnitude s.');
        end

        function testVeryLowTolPreservesMagnitude(tc)
            % A very strict tolerance must also preserve magnitude.
            [c_lt, ~, ~, ~] = rtpghifbwl(tc.s, tc.a(1), tc.fc, tc.tfr, 'tol', 1e-10);
            magErr = norm(abs(c_lt(:)) - tc.s(:)) / (norm(tc.s(:)) + eps);
            tc.verifyLessThan(magErr, 1e-6, ...
                'rtpghifbwl (low tol): |c| must still equal input magnitude s.');
        end

    end

    % ── Edge cases & robustness ───────────────────────────────────────────
    methods (Test)

        function testAllZeroMagnitudeInputReturnsZeroCoeffs(tc)
            % If s = 0 everywhere the reconstructed coefficients must also be zero.
            s_zero        = zeros(size(tc.s));
            [c_z, ~, ~, ~] = rtpghifbwl(s_zero, tc.a(1), tc.fc, tc.tfr);
            tc.verifyLessThan(norm(c_z(:)), 1e-14, ...
                'rtpghifbwl: all-zero magnitude must yield all-zero coefficients.');
        end

        function testSingleFrameInputWorks(tc)
            % A single-frame input (M x 1) must run without error.
            s_one = tc.s(:, 1);
            tc.verifyWarningFree( ...
                @() rtpghifbwl(s_one, tc.a(1), tc.fc, tc.tfr), ...
                'rtpghifbwl: single-frame input should work.');
            [c1, ~, ~, ~] = rtpghifbwl(s_one, tc.a(1), tc.fc, tc.tfr);
            tc.verifyEqual(size(c1), size(s_one), ...
                'rtpghifbwl: single-frame output size must match input size.');
        end

        function testPhaseIsFiniteEverywhere(tc)
            % Phase values must be finite (no NaN or Inf).
            [~, newphase, tgrad, fgrad] = rtpghifbwl(tc.s, tc.a(1), tc.fc, tc.tfr);
            tc.verifyTrue(all(isfinite(newphase(:))), ...
                'rtpghifbwl: newphase must be finite everywhere.');
            tc.verifyTrue(all(isfinite(tgrad(:))), ...
                'rtpghifbwl: tgrad must be finite everywhere.');
            tc.verifyTrue(all(isfinite(fgrad(:))), ...
                'rtpghifbwl: fgrad must be finite everywhere.');
        end

        function testCoefficientsAreFinite(tc)
            % Reconstructed coefficients must be finite.
            [c, ~, ~, ~] = rtpghifbwl(tc.s, tc.a(1), tc.fc, tc.tfr);
            tc.verifyTrue(all(isfinite(c(:))), ...
                'rtpghifbwl: c must be finite everywhere.');
        end

        function testImpulseSignalSmoke(tc)
            % An impulse signal (all energy in one sample) must run without error.
            imp    = tc.sig.impulse;
            c_imp  = ufilterbank(imp, tc.g, tc.a);
            s_imp  = abs(c_imp.');
            tc.verifyWarningFree( ...
                @() rtpghifbwl(s_imp, tc.a(1), tc.fc, tc.tfr), ...
                'rtpghifbwl: impulse signal must run without warnings.');
        end

    end

end
