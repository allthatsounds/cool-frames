classdef PropRtpghifbConsistency < matlab.unittest.TestCase
%PROPRTPGHIFBCONSISTENCY  Property tests for Real-Time Phase Gradient Heap
%                         Integration for filter banks (RTPGHIFB).
%
%   Verifies structural invariants of rtpghifbwl across multiple signals:
%
%   Magnitude invariance
%   --------------------
%   (1) |c(m,n)| == s(m,n) for all channels, all frames, all signals.
%   (2) Magnitude invariance holds for both 'normal' and 'causal' variants.
%
%   Phase consistency
%   -----------------
%   (3) c == s .* exp(1i * newphase) exactly (up to floating point).
%   (4) newphase is real-valued for any signal.
%   (5) Gradients tgrad and fgrad are real-valued for any signal.
%
%   Output structure
%   ----------------
%   (6) All four outputs have size M x N for all input signals.
%   (7) Coefficients c are finite everywhere.
%   (8) Phase newphase is finite and bounded (values in any real range are ok,
%       but NaN/Inf are not).
%
%   Reconstruction quality
%   ----------------------
%   (9) After ifilterbank + ufilterbank, the magnitude error (in dB) is
%       finite and strictly negative for multiple random signals.
%   (10) The 'causal' variant achieves a comparable (same sign) reconstruction
%        quality to the 'normal' variant.
%
%   Tolerance sensitivity
%   ---------------------
%   (11) Varying tol over a wide range [1e-10, 1e-2] does not break the
%        magnitude invariance property.
%
%   Gradient scaling
%   ----------------
%   (12) tgrad values are roughly on the scale of a * pi (they represent
%        phase advances per frame); they must not be identically zero.
%   (13) fgrad values must not be identically zero for a non-trivial signal.

    properties
        p       % parameter struct (fs, Ls, tol, abs_tol)
        g       % wavelet analysis filters
        a       % hop sizes (uniform)
        fc      % normalised centre frequencies
        tfr     % time-frequency ratio handle
        L       % system length
        M       % number of channels
        signals % struct of test signals from make_test_params
    end

    methods (TestClassSetup)
        function setupClass(tc)
            sharedDir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'shared');
            addpath(sharedDir);
            setup_filterbank_paths();
            rng(42);

            [tc.signals, tc.p] = make_test_params();
            Ls     = tc.p.Ls;
            scales = 2.^(linspace(5, -2, 64));
            [tc.g, tc.a, ~, tc.L, info] = ...
                waveletfilters(Ls, scales, 'repeat', 'uniform');
            tc.M   = numel(tc.g);
            tc.fc  = info.fc;
            tc.tfr = info.tfr;
        end
    end

    % ── Magnitude invariance ──────────────────────────────────────────────
    methods (Test)

        function testMagnitudeInvariantMultipleSignals(tc)
            % |c| == s must hold for 10 independent random signals.
            Ls = tc.p.Ls;
            rng(7);
            for trial = 1 : 10
                f      = randn(Ls, 1);
                corig  = ufilterbank(f, tc.g, tc.a);
                s      = abs(corig.');
                [c, ~, ~, ~] = rtpghifbwl(s, tc.a(1), tc.fc, tc.tfr);
                magErr = norm(abs(c(:)) - s(:)) / (norm(s(:)) + eps);
                tc.verifyLessThan(magErr, 1e-6, ...
                    sprintf('Trial %d: rtpghifbwl magnitude invariance violated.', trial));
            end
        end

        function testMagnitudeInvariantCausalVariant(tc)
            % The causal variant must also satisfy |c| == s for 10 signals.
            Ls = tc.p.Ls;
            rng(11);
            for trial = 1 : 10
                f     = randn(Ls, 1);
                corig = ufilterbank(f, tc.g, tc.a);
                s     = abs(corig.');
                [c, ~, ~, ~] = rtpghifbwl(s, tc.a(1), tc.fc, tc.tfr, 'causal');
                magErr = norm(abs(c(:)) - s(:)) / (norm(s(:)) + eps);
                tc.verifyLessThan(magErr, 1e-6, ...
                    sprintf('Trial %d (causal): magnitude invariance violated.', trial));
            end
        end

        function testMagnitudeInvariantBuiltInSignals(tc)
            % Test the standard signal set from make_test_params.
            signal_names = fieldnames(tc.signals);
            for k = 1 : numel(signal_names)
                fname = signal_names{k};
                f     = tc.signals.(fname);
                if ~isvector(f) || ~isreal(f)
                    continue;   % skip stereo / complex signals
                end
                f     = f(:);
                if numel(f) ~= tc.p.Ls
                    continue;   % length mismatch with filterbank
                end
                corig = ufilterbank(f, tc.g, tc.a);
                s     = abs(corig.');
                [c, ~, ~, ~] = rtpghifbwl(s, tc.a(1), tc.fc, tc.tfr);
                magErr = norm(abs(c(:)) - s(:)) / (norm(s(:)) + eps);
                tc.verifyLessThan(magErr, 1e-6, ...
                    sprintf('Signal "%s": magnitude invariance violated.', fname));
            end
        end

    end

    % ── Phase consistency ─────────────────────────────────────────────────
    methods (Test)

        function testCEqualsMAGTimesExpPhaseMultipleSignals(tc)
            % c == s .* exp(1i * newphase) must hold for 5 independent signals.
            Ls = tc.p.Ls;
            rng(13);
            for trial = 1 : 5
                f     = randn(Ls, 1);
                corig = ufilterbank(f, tc.g, tc.a);
                s     = abs(corig.');
                [c, newphase, ~, ~] = rtpghifbwl(s, tc.a(1), tc.fc, tc.tfr);
                c_expected = s .* exp(1i * newphase);
                relErr = norm(c(:) - c_expected(:)) / (norm(c_expected(:)) + eps);
                tc.verifyLessThan(relErr, 1e-10, ...
                    sprintf('Trial %d: c must equal s .* exp(1i * newphase).', trial));
            end
        end

        function testNewphaseIsRealForMultipleSignals(tc)
            % newphase must be real for all signals (it is an angle array).
            Ls = tc.p.Ls;
            rng(17);
            for trial = 1 : 10
                f     = randn(Ls, 1);
                corig = ufilterbank(f, tc.g, tc.a);
                s     = abs(corig.');
                [~, newphase, ~, ~] = rtpghifbwl(s, tc.a(1), tc.fc, tc.tfr);
                tc.verifyTrue(isreal(newphase), ...
                    sprintf('Trial %d: newphase must be real.', trial));
            end
        end

        function testGradientsAreRealForMultipleSignals(tc)
            % tgrad and fgrad must be real for all signals.
            Ls = tc.p.Ls;
            rng(19);
            for trial = 1 : 10
                f     = randn(Ls, 1);
                corig = ufilterbank(f, tc.g, tc.a);
                s     = abs(corig.');
                [~, ~, tgrad, fgrad] = rtpghifbwl(s, tc.a(1), tc.fc, tc.tfr);
                tc.verifyTrue(isreal(tgrad), ...
                    sprintf('Trial %d: tgrad must be real.', trial));
                tc.verifyTrue(isreal(fgrad), ...
                    sprintf('Trial %d: fgrad must be real.', trial));
            end
        end

    end

    % ── Output structure ──────────────────────────────────────────────────
    methods (Test)

        function testOutputSizeConsistentAcrossSignals(tc)
            % All four outputs must be (M x N) for every signal.
            Ls = tc.p.Ls;
            rng(23);
            for trial = 1 : 5
                f     = randn(Ls, 1);
                corig = ufilterbank(f, tc.g, tc.a);
                s     = abs(corig.');
                expectedSize = size(s);
                [c, newphase, tgrad, fgrad] = rtpghifbwl(s, tc.a(1), tc.fc, tc.tfr);
                tc.verifyEqual(size(c),        expectedSize, ...
                    sprintf('Trial %d: c size mismatch.', trial));
                tc.verifyEqual(size(newphase), expectedSize, ...
                    sprintf('Trial %d: newphase size mismatch.', trial));
                tc.verifyEqual(size(tgrad),    expectedSize, ...
                    sprintf('Trial %d: tgrad size mismatch.', trial));
                tc.verifyEqual(size(fgrad),    expectedSize, ...
                    sprintf('Trial %d: fgrad size mismatch.', trial));
            end
        end

        function testOutputIsFiniteForMultipleSignals(tc)
            % c, newphase, tgrad, fgrad must be finite for all signals.
            Ls = tc.p.Ls;
            rng(29);
            for trial = 1 : 10
                f     = randn(Ls, 1);
                corig = ufilterbank(f, tc.g, tc.a);
                s     = abs(corig.');
                [c, newphase, tgrad, fgrad] = rtpghifbwl(s, tc.a(1), tc.fc, tc.tfr);
                tc.verifyTrue(all(isfinite(c(:))),        ...
                    sprintf('Trial %d: c must be finite.', trial));
                tc.verifyTrue(all(isfinite(newphase(:))), ...
                    sprintf('Trial %d: newphase must be finite.', trial));
                tc.verifyTrue(all(isfinite(tgrad(:))),    ...
                    sprintf('Trial %d: tgrad must be finite.', trial));
                tc.verifyTrue(all(isfinite(fgrad(:))),    ...
                    sprintf('Trial %d: fgrad must be finite.', trial));
            end
        end

    end

    % ── Reconstruction quality ────────────────────────────────────────────
    methods (Test)

        function testReconstructionMagnitudeErrDbNegative(tc)
            % After ifilterbank + ufilterbank, magnitudeerrdb must be
            % finite and < 0 for all signals (better than random phase).
            Ls  = tc.p.Ls;
            gd  = filterbankrealdual(tc.g, tc.a, tc.L);
            rng(31);
            for trial = 1 : 5
                f     = randn(Ls, 1);
                corig = ufilterbank(f, tc.g, tc.a);
                s     = abs(corig.');
                [c, ~, ~, ~] = rtpghifbwl(s, tc.a(1), tc.fc, tc.tfr);
                f_rec  = ifilterbank(c.', gd, tc.a, 'real');
                c_rec  = ufilterbank(f_rec, tc.g, tc.a);
                errdb  = magnitudeerrdb(corig, c_rec);
                tc.verifyTrue(isfinite(errdb), ...
                    sprintf('Trial %d: magnitudeerrdb must be finite.', trial));
                tc.verifyLessThan(errdb, 0, ...
                    sprintf('Trial %d: magnitudeerrdb must be < 0 dB.', trial));
            end
        end

        function testCausalAndNormalBothAchieveNegativeErrDb(tc)
            % Both variants should achieve negative magnitude error dB.
            Ls  = tc.p.Ls;
            gd  = filterbankrealdual(tc.g, tc.a, tc.L);
            f   = tc.signals.noise_mono;
            corig = ufilterbank(f, tc.g, tc.a);
            s     = abs(corig.');

            for variant = {'normal', 'causal'}
                [c, ~, ~, ~] = rtpghifbwl(s, tc.a(1), tc.fc, tc.tfr, variant{1});
                f_rec  = ifilterbank(c.', gd, tc.a, 'real');
                c_rec  = ufilterbank(f_rec, tc.g, tc.a);
                errdb  = magnitudeerrdb(corig, c_rec);
                tc.verifyLessThan(errdb, 0, ...
                    sprintf('Variant "%s": magnitudeerrdb must be < 0 dB.', variant{1}));
            end
        end

    end

    % ── Tolerance sensitivity ─────────────────────────────────────────────
    methods (Test)

        function testMagnitudeInvariantAcrossTolerances(tc)
            % Magnitude invariance |c| == s must hold for a range of tol values.
            Ls     = tc.p.Ls;
            f      = tc.signals.noise_mono;
            corig  = ufilterbank(f, tc.g, tc.a);
            s      = abs(corig.');
            tols   = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2];
            for k = 1 : numel(tols)
                [c, ~, ~, ~] = rtpghifbwl(s, tc.a(1), tc.fc, tc.tfr, 'tol', tols(k));
                magErr = norm(abs(c(:)) - s(:)) / (norm(s(:)) + eps);
                tc.verifyLessThan(magErr, 1e-6, ...
                    sprintf('tol=%g: magnitude invariance violated.', tols(k)));
            end
        end

    end

    % ── Gradient scaling ──────────────────────────────────────────────────
    methods (Test)

        function testTgradNotIdenticallyZero(tc)
            % For a non-trivial signal, tgrad must contain non-zero values.
            f     = tc.signals.sine_440;
            corig = ufilterbank(f, tc.g, tc.a);
            s     = abs(corig.');
            [~, ~, tgrad, ~] = rtpghifbwl(s, tc.a(1), tc.fc, tc.tfr);
            tc.verifyGreaterThan(norm(tgrad(:)), 0, ...
                'rtpghifbwl: tgrad must not be identically zero for a sine signal.');
        end

        function testFgradNotIdenticallyZero(tc)
            % For a non-trivial signal, fgrad must contain non-zero values.
            f     = tc.signals.noise_mono;
            corig = ufilterbank(f, tc.g, tc.a);
            s     = abs(corig.');
            [~, ~, ~, fgrad] = rtpghifbwl(s, tc.a(1), tc.fc, tc.tfr);
            tc.verifyGreaterThan(norm(fgrad(:)), 0, ...
                'rtpghifbwl: fgrad must not be identically zero for a noise signal.');
        end

        function testTgradScaleIsConsistentWithHopSize(tc)
            % tgrad represents phase advance per frame; its RMS should be
            % on the order of a*pi (not orders of magnitude off).
            f     = tc.signals.noise_mono;
            corig = ufilterbank(f, tc.g, tc.a);
            s     = abs(corig.');
            [~, ~, tgrad, ~] = rtpghifbwl(s, tc.a(1), tc.fc, tc.tfr);
            rms_tgrad = rms(tgrad(:));
            % Accept anything between a*pi/100 and a*pi*100 as "reasonable".
            expected_scale = tc.a(1) * pi;
            tc.verifyGreaterThan(rms_tgrad, expected_scale / 100, ...
                'rtpghifbwl: tgrad RMS is unexpectedly small.');
            tc.verifyLessThan(rms_tgrad, expected_scale * 100, ...
                'rtpghifbwl: tgrad RMS is unexpectedly large.');
        end

    end

end
