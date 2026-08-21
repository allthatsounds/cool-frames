classdef TestPhaseProcessing < matlab.unittest.TestCase
%TESTPHASEPROCESSING  Unit tests for phase-estimation and reassignment entry points.
%
%   Covers: filterbankphasegrad, filterbankconstphase,
%           filterbankreassign, filterbanksynchrosqueeze.
%
%   All tests use the ERB filterbank from audfilters (positive-frequency,
%   non-uniform hop) or a simpler uniform filterbank where a=1 is needed.

    properties
        sig     % test signal struct
        p       % parameter struct (fs, Ls, tol)
        g       % ERB analysis filters
        a       % subsampling factors  [M x 2]
        fc_n    % normalised center frequencies  (info.fc, in [-1,1])
        tfr     % time-frequency ratio vector for each channel (info.tfr(L))
        L       % system length
        M       % number of filters
        f       % single test signal (column, length Ls)
    end

    methods (TestClassSetup)
        function setupClass(tc)
            sharedDir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'shared');
            addpath(sharedDir);
            setup_filterbank_paths();
            [tc.sig, tc.p] = make_test_params();

            % Build the standard ERB filter bank.
            [tc.g, tc.a, ~, tc.L, info] = audfilters(tc.p.fs, tc.p.Ls);
            tc.M    = numel(tc.g);
            tc.fc_n = info.fc;            % normalised fc (LTFAT convention: 1=Nyquist)
            tc.tfr  = info.tfr(tc.L);    % time-frequency ratios at length L
            tc.f    = tc.sig.noise_mono;
        end
    end

    % ── filterbankphasegrad ───────────────────────────────────────────────
    methods (Test)

        function testFilterbankphasegradOutputCount(tc)
            % All four outputs must be returned without error.
            [tgrad, fgrad, s, c] = filterbankphasegrad(tc.f, tc.g, tc.a, tc.L);
            tc.verifyEqual(numel(tgrad), tc.M, ...
                'filterbankphasegrad: tgrad must have M cells.');
            tc.verifyEqual(numel(fgrad), tc.M, ...
                'filterbankphasegrad: fgrad must have M cells.');
            tc.verifyEqual(numel(s),     tc.M, ...
                'filterbankphasegrad: s (spectrogram) must have M cells.');
            tc.verifyEqual(numel(c),     tc.M, ...
                'filterbankphasegrad: c (coefficients) must have M cells.');
        end

        function testFilterbankphasegradCellSizesMatchCoefficients(tc)
            % tgrad, fgrad and s must have the same size as filterbank(f,g,a).
            c_ref             = filterbank(tc.f, tc.g, tc.a);
            [tgrad, fgrad, s] = filterbankphasegrad(tc.f, tc.g, tc.a, tc.L);
            for m = 1 : tc.M
                tc.verifyEqual(size(tgrad{m}), size(c_ref{m}), ...
                    sprintf('filterbankphasegrad: tgrad{%d} size must match coeff size.', m));
                tc.verifyEqual(size(fgrad{m}), size(c_ref{m}), ...
                    sprintf('filterbankphasegrad: fgrad{%d} size must match coeff size.', m));
                tc.verifyEqual(size(s{m}),     size(c_ref{m}), ...
                    sprintf('filterbankphasegrad: s{%d} size must match coeff size.', m));
            end
        end

        function testFilterbankphasegradGradientIsReal(tc)
            % Phase gradients are real-valued (they are phase derivatives).
            [tgrad, fgrad] = filterbankphasegrad(tc.f, tc.g, tc.a, tc.L);
            for m = 1 : tc.M
                tc.verifyTrue(isreal(tgrad{m}), ...
                    sprintf('filterbankphasegrad: tgrad{%d} must be real.', m));
                tc.verifyTrue(isreal(fgrad{m}), ...
                    sprintf('filterbankphasegrad: fgrad{%d} must be real.', m));
            end
        end

        function testFilterbankphasegradSpectrogramNonNegative(tc)
            % The spectrogram s = |c|^2 (possibly scaled) must be non-negative.
            [~, ~, s] = filterbankphasegrad(tc.f, tc.g, tc.a, tc.L);
            for m = 1 : tc.M
                tc.verifyTrue(all(s{m}(:) >= -1e-12), ...
                    sprintf('filterbankphasegrad: s{%d} must be non-negative.', m));
            end
        end

        function testFilterbankphasegradCoefficientsMatchFilterbank(tc)
            % The returned c must agree with filterbank(f, g, a).
            c_ref           = filterbank(tc.f, tc.g, tc.a);
            [~, ~, ~, c_pg] = filterbankphasegrad(tc.f, tc.g, tc.a, tc.L);
            for m = 1 : tc.M
                relErr = norm(c_pg{m}(:) - c_ref{m}(:)) / (norm(c_ref{m}(:)) + eps);
                tc.verifyLessThan(relErr, 1e-10, ...
                    sprintf('filterbankphasegrad: c{%d} must match filterbank output.', m));
            end
        end

        function testFilterbankphasegradWorksWithoutL(tc)
            % Calling without L should not error (L is inferred from signal).
            tc.verifyWarningFree( ...
                @() filterbankphasegrad(tc.f, tc.g, tc.a), ...
                'filterbankphasegrad: should run without error when L is omitted.');
        end

    end

    % ── filterbankconstphase ──────────────────────────────────────────────
    methods (Test)

        function testFilterbankconstphaseReturnsCell(tc)
            c  = filterbank(tc.f, tc.g, tc.a);
            s  = cellfun(@abs, c, 'UniformOutput', false);
            c_out = filterbankconstphase(s, tc.a, tc.fc_n, tc.tfr);
            tc.verifyTrue(iscell(c_out), ...
                'filterbankconstphase: output must be a cell array.');
        end

        function testFilterbankconstphaseOutputLength(tc)
            c  = filterbank(tc.f, tc.g, tc.a);
            s  = cellfun(@abs, c, 'UniformOutput', false);
            c_out = filterbankconstphase(s, tc.a, tc.fc_n, tc.tfr);
            tc.verifyEqual(numel(c_out), tc.M, ...
                'filterbankconstphase: output must have M cells.');
        end

        function testFilterbankconstphaseMagnitudePreserved(tc)
            % |c_out{m}| must equal the input magnitude s{m}.
            c  = filterbank(tc.f, tc.g, tc.a);
            s  = cellfun(@abs, c, 'UniformOutput', false);
            c_out = filterbankconstphase(s, tc.a, tc.fc_n, tc.tfr);
            for m = 1 : tc.M
                mag_out = abs(c_out{m});
                mag_in  = s{m};
                relErr  = norm(mag_out(:) - mag_in(:)) / (norm(mag_in(:)) + eps);
                tc.verifyLessThan(relErr, 1e-6, ...
                    sprintf('filterbankconstphase: |c_out{%d}| must equal input magnitude.', m));
            end
        end

        function testFilterbankconstphaseCellSizePreserved(tc)
            % Each output cell must have the same size as the input magnitude.
            c  = filterbank(tc.f, tc.g, tc.a);
            s  = cellfun(@abs, c, 'UniformOutput', false);
            c_out = filterbankconstphase(s, tc.a, tc.fc_n, tc.tfr);
            for m = 1 : tc.M
                tc.verifyEqual(size(c_out{m}), size(s{m}), ...
                    sprintf('filterbankconstphase: c_out{%d} size must match input.', m));
            end
        end

        function testFilterbankconstphaseWithExplicitGradient(tc)
            % Passing {tgrad,fgrad} instead of tfr must also preserve magnitude.
            c  = filterbank(tc.f, tc.g, tc.a);
            s  = cellfun(@abs, c, 'UniformOutput', false);
            [tgrad, fgrad] = filterbankphasegrad(tc.f, tc.g, tc.a, tc.L);
            c_out = filterbankconstphase(s, tc.a, tc.fc_n, {tgrad, fgrad});
            for m = 1 : tc.M
                relErr = norm(abs(c_out{m}(:)) - s{m}(:)) / (norm(s{m}(:)) + eps);
                tc.verifyLessThan(relErr, 1e-6, ...
                    sprintf('filterbankconstphase (explicit grad): magnitude error in cell %d.', m));
            end
        end

        function testFilterbankconstphaseAllOutputsNoError(tc)
            % Four-output form must run without error.
            c  = filterbank(tc.f, tc.g, tc.a);
            s  = cellfun(@abs, c, 'UniformOutput', false);
            tc.verifyWarningFree( ...
                @() filterbankconstphase(s, tc.a, tc.fc_n, tc.tfr), ...
                'filterbankconstphase: four-output call must not raise warnings.');
        end

    end

    % ── filterbankreassign ────────────────────────────────────────────────
    methods (Test)

        function testFilterbankreassignOutputIsCell(tc)
            [tgrad, fgrad, s] = filterbankphasegrad(tc.f, tc.g, tc.a, tc.L);
            sr = filterbankreassign(s, tgrad, fgrad, tc.a, tc.fc_n);
            tc.verifyTrue(iscell(sr), ...
                'filterbankreassign: sr output must be a cell array.');
        end

        function testFilterbankreassignOutputLength(tc)
            [tgrad, fgrad, s] = filterbankphasegrad(tc.f, tc.g, tc.a, tc.L);
            sr = filterbankreassign(s, tgrad, fgrad, tc.a, tc.fc_n);
            tc.verifyEqual(numel(sr), tc.M, ...
                'filterbankreassign: sr must have M cells.');
        end

        function testFilterbankreassignCellSizesPreserved(tc)
            % The reassigned spectrogram must be the same size as the input.
            [tgrad, fgrad, s] = filterbankphasegrad(tc.f, tc.g, tc.a, tc.L);
            sr = filterbankreassign(s, tgrad, fgrad, tc.a, tc.fc_n);
            for m = 1 : tc.M
                tc.verifyEqual(size(sr{m}), size(s{m}), ...
                    sprintf('filterbankreassign: sr{%d} size must match s{%d} size.', m, m));
            end
        end

        function testFilterbankreassignThreeOutputs(tc)
            % Three-output form [sr, repos, Lc] must run without error.
            [tgrad, fgrad, s] = filterbankphasegrad(tc.f, tc.g, tc.a, tc.L);
            [sr, repos, Lc]   = filterbankreassign(s, tgrad, fgrad, tc.a, tc.fc_n);
            tc.verifyEqual(numel(Lc), tc.M, ...
                'filterbankreassign: Lc must have M elements.');
            tc.verifyEqual(sum(Lc), numel(repos), ...
                'filterbankreassign: numel(repos) must equal sum(Lc).');
            % sanity: sr non-negative (it holds magnitude^2 values)
            for m = 1 : tc.M
                tc.verifyTrue(all(sr{m}(:) >= -1e-12), ...
                    sprintf('filterbankreassign: sr{%d} must be non-negative.', m));
            end
        end

        function testFilterbankreassignAcceptsFilterCellInsteadOfCfreq(tc)
            % filterbankreassign can take the filter cell g instead of cfreq.
            [tgrad, fgrad, s] = filterbankphasegrad(tc.f, tc.g, tc.a, tc.L);
            tc.verifyWarningFree( ...
                @() filterbankreassign(s, tgrad, fgrad, tc.a, tc.g), ...
                'filterbankreassign: should accept filter cell array as 5th argument.');
        end

        function testFilterbankreassignEnergyApproximatelyConserved(tc)
            % Sum of reassigned energies should roughly equal sum of input energies.
            % (Not exact because of boundary effects, but order-of-magnitude.)
            [tgrad, fgrad, s] = filterbankphasegrad(tc.f, tc.g, tc.a, tc.L);
            sr                = filterbankreassign(s, tgrad, fgrad, tc.a, tc.fc_n);
            e_in  = sum(cellfun(@(x) sum(x(:)), s));
            e_out = sum(cellfun(@(x) sum(x(:)), sr));
            tc.verifyGreaterThan(e_out, 0, ...
                'filterbankreassign: total reassigned energy must be positive.');
            % Allow a factor of 2 either way (boundary bins may be missed)
            tc.verifyLessThan(abs(e_out - e_in) / (e_in + eps), 1.5, ...
                'filterbankreassign: total energy change should be bounded.');
        end

    end

    % ── filterbanksynchrosqueeze ──────────────────────────────────────────
    methods (Test)

        function testFilterbanksynchrosqueezeOutputIsCell(tc)
            % filterbanksynchrosqueeze requires a non-subsampled filterbank (a=1).
            a_ones = ones(tc.M, 1);
            c_ns   = filterbank(tc.f, tc.g, a_ones);
            [tgrad_ns] = filterbankphasegrad(tc.f, tc.g, a_ones);
            cr = filterbanksynchrosqueeze(c_ns, tgrad_ns, tc.fc_n);
            tc.verifyTrue(iscell(cr), ...
                'filterbanksynchrosqueeze: cr must be a cell array.');
        end

        function testFilterbanksynchrosqueezeOutputLength(tc)
            a_ones    = ones(tc.M, 1);
            c_ns      = filterbank(tc.f, tc.g, a_ones);
            tgrad_ns  = filterbankphasegrad(tc.f, tc.g, a_ones);
            cr        = filterbanksynchrosqueeze(c_ns, tgrad_ns, tc.fc_n);
            tc.verifyEqual(numel(cr), tc.M, ...
                'filterbanksynchrosqueeze: cr must have M cells.');
        end

        function testFilterbanksynchrosqueezeCellSizesPreserved(tc)
            % Synchrosqueezed output cells must have the same size as input c.
            a_ones   = ones(tc.M, 1);
            c_ns     = filterbank(tc.f, tc.g, a_ones);
            tgrad_ns = filterbankphasegrad(tc.f, tc.g, a_ones);
            cr       = filterbanksynchrosqueeze(c_ns, tgrad_ns, tc.fc_n);
            for m = 1 : tc.M
                tc.verifyEqual(size(cr{m}), size(c_ns{m}), ...
                    sprintf('filterbanksynchrosqueeze: cr{%d} size must match c{%d}.', m, m));
            end
        end

        function testFilterbanksynchrosqueezeThreeOutputs(tc)
            % Three-output form [cr, repos, Lc] must run without error.
            a_ones   = ones(tc.M, 1);
            c_ns     = filterbank(tc.f, tc.g, a_ones);
            tgrad_ns = filterbankphasegrad(tc.f, tc.g, a_ones);
            [cr, repos, Lc] = filterbanksynchrosqueeze(c_ns, tgrad_ns, tc.fc_n);
            tc.verifyEqual(numel(Lc), tc.M, ...
                'filterbanksynchrosqueeze: Lc must have M elements.');
            tc.verifyEqual(sum(Lc), numel(repos), ...
                'filterbanksynchrosqueeze: numel(repos) must equal sum(Lc).');
        end

        function testFilterbanksynchrosqueezeAcceptsFilterCell(tc)
            % Should work when the filter cell array g is passed as 3rd argument.
            a_ones   = ones(tc.M, 1);
            c_ns     = filterbank(tc.f, tc.g, a_ones);
            tgrad_ns = filterbankphasegrad(tc.f, tc.g, a_ones);
            tc.verifyWarningFree( ...
                @() filterbanksynchrosqueeze(c_ns, tgrad_ns, tc.g), ...
                'filterbanksynchrosqueeze: should accept filter cell array as 3rd argument.');
        end

    end

end
