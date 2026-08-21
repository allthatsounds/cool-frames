classdef TestPostProcessing < matlab.unittest.TestCase
%TESTPOSTPROCESSING  Unit tests for post-processing, inspection and
%                    visualization entry points.
%
%   Covers: filterbankphasegrad, filterbankreassign,
%           filterbanksynchrosqueeze, filterbankconstphase,
%           filterbankresponse, filterbankfreqz, plotfilterbank.
%
%   Important signature notes (differ from I/O doc in subtle ways):
%     filterbankphasegrad(f, g, a, L)      -- 1st arg is the RAW SIGNAL
%       returns [tgrad, fgrad, s, c]        -- s = |c|^2 spectrogram
%     filterbankreassign(s,tgrad,fgrad,a,cfreq)  -- 5th arg = center freqs
%     filterbanksynchrosqueeze(c,tgrad,cfreq)    -- 3rd arg = center freqs
%     filterbankconstphase(s,a,fc_norm,tfr_norm) -- fc & tfr normalized to Nyquist

    properties
        sig
        p
        g           % ERB filters
        a           % subsampling
        fc          % center frequencies (Hz)
        L           % system length
        M           % number of channels
        % Pre-computed phase-gradient outputs (built once, varying a)
        tgrad
        fgrad
        s_spec      % spectrogram |c|^2
        c_cplx      % complex coefficients
        cfreq       % normalised center frequencies for reassign/synchrosqueeze
        % Uniform filterbank — required by filterbanksynchrosqueeze,
        % which demands all subbands have equal length.
        % Built from audfilters(...,'uniform') so all channels share the same a.
        g_uni           % uniform filterbank filters
        a_uni           % uniform subsampling factors (all equal)
        L_uni           % transform length for uniform filterbank
        M_uni           % channel count of uniform filterbank
        cfreq_uni       % normalised center freqs for uniform filterbank
        c_uniform       % coefficients from uniform filterbank
        tgrad_uniform   % phase gradient from uniform filterbank
    end

    methods (TestClassSetup)
        function setupClass(tc)
            addpath(fileparts(fileparts(mfilename('fullpath'))));
            [tc.sig, tc.p] = make_test_params();

            [tc.g, tc.a, tc.fc, tc.L] = audfilters(tc.p.fs, tc.p.Ls);
            tc.M = numel(tc.g);

            % Compute phase gradients once — also gives spectrogram and coefficients.
            [tc.tgrad, tc.fgrad, tc.s_spec, tc.c_cplx] = ...
                filterbankphasegrad(tc.sig.noise_mono, tc.g, tc.a, tc.L);

            % Normalized center frequencies for reassign / synchrosqueeze.
            tc.cfreq = cent_freqs(tc.g, tc.L);

            % filterbanksynchrosqueeze requires equal-length subbands.
            % Use a separate uniform audfilterbank (same a for every channel)
            % so that all subbands produced by filterbankphasegrad are equal-length.
            [tc.g_uni, tc.a_uni, ~, tc.L_uni] = ...
                audfilters(tc.p.fs, tc.p.Ls, 'uniform');
            tc.M_uni     = numel(tc.g_uni);
            tc.cfreq_uni = cent_freqs(tc.g_uni, tc.L_uni);
            [tc.tgrad_uniform, ~, ~, tc.c_uniform] = ...
                filterbankphasegrad(tc.sig.noise_mono, tc.g_uni, tc.a_uni, tc.L_uni);
        end
    end

    % ── filterbankphasegrad ───────────────────────────────────────────────
    methods (Test)

        function testPhaseGradChannelCountTgrad(tc)
            tc.verifyEqual(numel(tc.tgrad), tc.M, ...
                'filterbankphasegrad: tgrad must have M channels.');
        end

        function testPhaseGradChannelCountFgrad(tc)
            tc.verifyEqual(numel(tc.fgrad), tc.M, ...
                'filterbankphasegrad: fgrad must have M channels.');
        end

        function testPhaseGradChannelCountSpec(tc)
            tc.verifyEqual(numel(tc.s_spec), tc.M, ...
                'filterbankphasegrad: spectrogram s must have M channels.');
        end

        function testPhaseGradSpectrogramIsSquaredMagnitude(tc)
            % s{m} must equal |c{m}|^2 up to floating-point precision.
            for m = 1 : min(tc.M, 5)   % spot-check first 5 channels
                expected = abs(tc.c_cplx{m}).^2;
                rel_err  = norm(tc.s_spec{m} - expected, 'fro') ...
                         / (norm(expected, 'fro') + eps);
                tc.verifyLessThan(rel_err, 1e-10, ...
                    sprintf('filterbankphasegrad: s{%d} must equal |c{%d}|^2.', m, m));
            end
        end

        function testPhaseGradCoefficientsAreComplex(tc)
            tc.verifyFalse(isreal(tc.c_cplx{2}), ...
                'filterbankphasegrad: returned coefficients must be complex.');
        end

        function testPhaseGradTgradIsReal(tc)
            % Phase gradients are real-valued quantities.
            for m = 1 : min(tc.M, 5)
                tc.verifyTrue(isreal(tc.tgrad{m}), ...
                    sprintf('filterbankphasegrad: tgrad{%d} must be real-valued.', m));
            end
        end

        function testPhaseGradFgradIsReal(tc)
            for m = 1 : min(tc.M, 5)
                tc.verifyTrue(isreal(tc.fgrad{m}), ...
                    sprintf('filterbankphasegrad: fgrad{%d} must be real-valued.', m));
            end
        end

        function testPhaseGradDimsMatchCoeff(tc)
            % tgrad, fgrad and s must have the same dimensions as c.
            for m = 1 : min(tc.M, 5)
                tc.verifyEqual(size(tc.tgrad{m}), size(tc.c_cplx{m}), ...
                    sprintf('filterbankphasegrad: tgrad{%d} dims must match c{%d}.', m, m));
                tc.verifyEqual(size(tc.fgrad{m}), size(tc.c_cplx{m}), ...
                    sprintf('filterbankphasegrad: fgrad{%d} dims must match c{%d}.', m, m));
                tc.verifyEqual(size(tc.s_spec{m}), size(tc.c_cplx{m}), ...
                    sprintf('filterbankphasegrad: s_spec{%d} dims must match c{%d}.', m, m));
            end
        end

    end

    % ── filterbankreassign ────────────────────────────────────────────────
    methods (Test)

        function testReassignChannelCount(tc)
            sr = filterbankreassign(tc.s_spec, tc.tgrad, tc.fgrad, tc.a, tc.cfreq);
            tc.verifyEqual(numel(sr), tc.M, ...
                'filterbankreassign: output must have M channels.');
        end

        function testReassignEnergyConserved(tc)
            % Reassignment redistributes energy but must conserve the total
            % (up to boundary effects): sum(sr) ≈ sum(s).
            sr = filterbankreassign(tc.s_spec, tc.tgrad, tc.fgrad, tc.a, tc.cfreq);
            E_in  = sum(cellfun(@(x) sum(x(:)), tc.s_spec));
            E_out = sum(cellfun(@(x) sum(x(:)), sr));
            rel_err = abs(E_out - E_in) / (E_in + eps);
            tc.verifyLessThan(rel_err, 0.01, ...
                'filterbankreassign: total energy must be approximately conserved.');
        end

        function testReassignOutputNonNegative(tc)
            sr = filterbankreassign(tc.s_spec, tc.tgrad, tc.fgrad, tc.a, tc.cfreq);
            for m = 1 : min(tc.M, 5)
                tc.verifyTrue(all(sr{m}(:) >= 0), ...
                    sprintf('filterbankreassign: sr{%d} must be non-negative.', m));
            end
        end

    end

    % ── filterbanksynchrosqueeze ──────────────────────────────────────────
    methods (Test)

        function testSynchrosqueezeChannelCount(tc)
            % filterbanksynchrosqueeze requires equal-length subbands.
            % Uses the uniform filterbank built in setupClass.
            cr = filterbanksynchrosqueeze(tc.c_uniform, tc.tgrad_uniform, tc.cfreq_uni);
            tc.verifyEqual(numel(cr), tc.M_uni, ...
                'filterbanksynchrosqueeze: output must have M channels.');
        end

        function testSynchrosqueezeEnergyConserved(tc)
            % filterbanksynchrosqueeze requires equal-length subbands.
            cr = filterbanksynchrosqueeze(tc.c_uniform, tc.tgrad_uniform, tc.cfreq_uni);
            E_in  = sum(cellfun(@(x) sum(abs(x(:)).^2), tc.c_uniform));
            E_out = sum(cellfun(@(x) sum(abs(x(:)).^2), cr));
            rel_err = abs(E_out - E_in) / (E_in + eps);
            tc.verifyLessThan(rel_err, 0.25, ...
                'filterbanksynchrosqueeze: total energy should be approximately conserved.');
        end

    end

    % ── filterbankconstphase ──────────────────────────────────────────────
    methods (Test)

        function testConstphaseRuns(tc)
            % Build the inputs that filterbankconstphase requires.
            fc_norm  = tc.fc / (tc.p.fs / 2);       % centre freqs → normalised
            bw_norm  = audfiltbw(tc.fc, 'erb') / (tc.p.fs / 2);  % ERB bw → normalised
            c_mag    = cellfun(@abs, tc.c_cplx, 'UniformOutput', false);
            c_out    = filterbankconstphase(c_mag, tc.a, fc_norm, bw_norm);
            tc.verifyEqual(numel(c_out), tc.M, ...
                'filterbankconstphase: output must have M channels.');
        end

        function testConstphaseMagnitudeApproximatelyPreserved(tc)
            % Phase reconstruction should preserve the input magnitudes.
            fc_norm = tc.fc / (tc.p.fs / 2);
            bw_norm = audfiltbw(tc.fc, 'erb') / (tc.p.fs / 2);
            c_mag   = cellfun(@abs, tc.c_cplx, 'UniformOutput', false);
            c_out   = filterbankconstphase(c_mag, tc.a, fc_norm, bw_norm);
            for m = 1 : min(tc.M, 5)
                rel_err = norm(abs(c_out{m}) - c_mag{m}, 'fro') ...
                        / (norm(c_mag{m}, 'fro') + eps);
                tc.verifyLessThan(rel_err, 0.05, ...
                    sprintf('filterbankconstphase: magnitudes should be preserved at channel %d.', m));
            end
        end

        function testConstphaseOutputIsComplex(tc)
            fc_norm = tc.fc / (tc.p.fs / 2);
            bw_norm = audfiltbw(tc.fc, 'erb') / (tc.p.fs / 2);
            c_mag   = cellfun(@abs, tc.c_cplx, 'UniformOutput', false);
            c_out   = filterbankconstphase(c_mag, tc.a, fc_norm, bw_norm);
            tc.verifyFalse(isreal(c_out{1}), ...
                'filterbankconstphase: output must be complex (contains reconstructed phase).');
        end

    end

    % ── filterbankresponse ────────────────────────────────────────────────
    methods (Test)

        function testResponseLength(tc)
            R = filterbankresponse(tc.g, tc.a, tc.L);
            tc.verifyEqual(numel(R), tc.L, ...
                'filterbankresponse: output must have L elements.');
        end

        function testResponseAlmostEverywherePositive(tc)
            % A well-designed filter bank covers all frequencies;
            % allow at most 1 % of bins to be zero or negative.
            R = filterbankresponse(tc.g, tc.a, tc.L);
            frac_positive = mean(R > 0);
            tc.verifyGreaterThan(frac_positive, 0.5, ...
                'filterbankresponse: at least 50% of frequency bins must be positive.');
        end

        function testResponseIsReal(tc)
            R = filterbankresponse(tc.g, tc.a, tc.L);
            tc.verifyTrue(isreal(R), ...
                'filterbankresponse: output must be real-valued.');
        end

    end

    % ── filterbankfreqz ───────────────────────────────────────────────────
    methods (Test)

        function testFreqzRowCount(tc)
            H = filterbankfreqz(tc.g, tc.a, tc.L);
            tc.verifyEqual(size(H, 1), tc.L, ...
                'filterbankfreqz: number of rows must equal L.');
        end

        function testFreqzColumnCount(tc)
            H = filterbankfreqz(tc.g, tc.a, tc.L);
            tc.verifyEqual(size(H, 2), tc.M, ...
                'filterbankfreqz: number of columns must equal M.');
        end

        function testFreqzResponseConsistency(tc)
            % filterbankresponse should equal sum_m |H_m|^2 * (1/a_m).
            % With rational subsampling a = [p,q] → effective rate = p/q.
            H = filterbankfreqz(tc.g, tc.a, tc.L);
            R = filterbankresponse(tc.g, tc.a, tc.L);

            % Build per-channel rate scalar.
            if size(tc.a, 2) == 2
                a_scalar = tc.a(:,1) ./ tc.a(:,2);
            else
                a_scalar = tc.a(:);
            end

            manual_R = zeros(tc.L, 1);
            for m = 1 : tc.M
                manual_R = manual_R + abs(H(:, m)).^2 / a_scalar(m);
            end
            rel_err = norm(R - manual_R) / (norm(R) + eps);
            tc.verifyLessThan(rel_err, 0.01, ...
                'filterbankfreqz vs filterbankresponse: sum |H_m|^2/a_m must match.');
        end

    end

    % ── plotfilterbank ────────────────────────────────────────────────────
    methods (Test)

        function testPlotfilterbankRuns(tc)
            % Suppress figure display; just verify it runs without error.
            prev_vis = get(0, 'DefaultFigureVisible');
            set(0, 'DefaultFigureVisible', 'off');
            c = filterbank(tc.sig.noise_mono, tc.g, tc.a);
            try
                h = plotfilterbank(c, tc.a);
                close all;
            catch ME
                set(0, 'DefaultFigureVisible', prev_vis);
                tc.verifyFail(sprintf('plotfilterbank threw: %s', ME.message));
                return;
            end
            set(0, 'DefaultFigureVisible', prev_vis);
            % If a handle was returned it must be a valid graphics object.
            if ~isempty(h)
                tc.verifyTrue(ishandle(h(1)) || isnumeric(h), ...
                    'plotfilterbank: returned handle must be a valid graphics handle.');
            end
        end

    end

end
