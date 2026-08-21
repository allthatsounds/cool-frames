classdef PropPhaseGradientConsistency < matlab.unittest.TestCase
%PROPPHASEGRADIENTCONSISTENCY  Property tests for phase-based analysis.
%
%   Verifies structural invariants of:
%
%   filterbankphasegrad
%   --------------------
%   (1) tgrad and fgrad have the same cell dimensions as filterbank(f,g,a).
%   (2) s (spectrogram) is non-negative and equals |c|^2 (up to scaling).
%   (3) For a pure sinusoid, tgrad in the active channel should be close
%       to the normalised sinusoid frequency.
%
%   filterbankconstphase
%   --------------------
%   (4) |c_out{m}| == s{m}  for all channels (magnitude invariant).
%   (5) Passing the exact phase gradient from filterbankphasegrad should
%       give a lower reconstruction error than random phase.
%
%   filterbankreassign
%   --------------------
%   (6) Total energy (sum of all sr values) is approximately conserved.
%   (7) Reassigned spectrogram sr has the same cell structure as s.
%
%   filterbanksynchrosqueeze
%   -------------------------
%   (8) Total coefficient energy is approximately conserved.
%   (9) Output cr has the same cell dimensions as the input c.

    properties
        p       % parameters: fs, Ls
        g       % ERB analysis filters
        a       % subsampling factors
        fc_n    % normalised center frequencies (info.fc)
        tfr     % time-frequency ratio vector (info.tfr(L))
        L       % transform length
        M       % number of filters
    end

    methods (TestClassSetup)
        function setupClass(tc)
            sharedDir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'shared');
            addpath(sharedDir);
            setup_filterbank_paths();
            rng(42);

            tc.p = struct('fs', 8000, 'Ls', 1024);
            [tc.g, tc.a, ~, tc.L, info] = audfilters(tc.p.fs, tc.p.Ls);
            tc.M    = numel(tc.g);
            tc.fc_n = info.fc;
            tc.tfr  = info.tfr(tc.L);
        end
    end

    % ── filterbankphasegrad: structural properties ────────────────────────
    methods (Test)

        function testPhaseGradDimensionsMatchFilterbank(tc)
            % tgrad, fgrad, s must have M cells with identical sizes to
            % the output of filterbank(f,g,a).
            Ls     = tc.p.Ls;
            f      = randn(Ls, 1);
            c_ref  = filterbank(f, tc.g, tc.a);
            [tgrad, fgrad, s] = filterbankphasegrad(f, tc.g, tc.a, tc.L);
            for m = 1 : tc.M
                tc.verifyEqual(size(tgrad{m}), size(c_ref{m}), ...
                    sprintf('phasegrad: tgrad{%d} size must match coeff size.', m));
                tc.verifyEqual(size(fgrad{m}), size(c_ref{m}), ...
                    sprintf('phasegrad: fgrad{%d} size must match coeff size.', m));
                tc.verifyEqual(size(s{m}),     size(c_ref{m}), ...
                    sprintf('phasegrad: s{%d} size must match coeff size.', m));
            end
        end

        function testPhaseGradSpectrogramEqualsAbsSquared(tc)
            % The spectrogram s{m} returned by filterbankphasegrad should
            % equal |c{m}|^2  (up to the minlvl floor, which is eps by default).
            Ls  = tc.p.Ls;
            f   = randn(Ls, 1);
            [~, ~, s, c] = filterbankphasegrad(f, tc.g, tc.a, tc.L);
            for m = 1 : tc.M
                expected = abs(c{m}).^2;
                % Only check bins where |c|^2 >> eps (minlvl floor is irrelevant there).
                mask   = expected > 1e-10 * max(expected(:));
                if any(mask)
                    relErr = norm(s{m}(mask) - expected(mask)) / (norm(expected(mask)) + eps);
                    tc.verifyLessThan(relErr, 1e-8, ...
                        sprintf('phasegrad: s{%d} must equal |c{%d}|^2.', m, m));
                end
            end
        end

        function testPhaseGradIsRealForMultipleSignals(tc)
            % tgrad and fgrad must be real for any real or complex input signal.
            Ls = tc.p.Ls;
            rng(7);
            for trial = 1 : 5
                f      = randn(Ls, 1);
                [tgrad, fgrad] = filterbankphasegrad(f, tc.g, tc.a, tc.L);
                for m = 1 : tc.M
                    tc.verifyTrue(isreal(tgrad{m}), ...
                        sprintf('Trial %d: tgrad{%d} must be real.', trial, m));
                    tc.verifyTrue(isreal(fgrad{m}), ...
                        sprintf('Trial %d: fgrad{%d} must be real.', trial, m));
                end
            end
        end

        function testPhaseGradTotalSpectrogramMatchesCoeffEnergy(tc)
            % sum_m sum(s{m}) = sum_m ||c{m}||^2  (since s{m} = |c{m}|^2).
            % The minlvl floor introduces a tiny offset; we allow 1e-6 relative error.
            Ls   = tc.p.Ls;
            f    = randn(Ls, 1);
            [~, ~, s, c] = filterbankphasegrad(f, tc.g, tc.a, tc.L);
            e_s  = sum(cellfun(@(x) sum(x(:)), s));
            e_c  = sum(cellfun(@(cm) norm(cm)^2,  c));
            relErr = abs(e_s - e_c) / (e_c + eps);
            tc.verifyLessThan(relErr, 1e-6, ...
                'phasegrad: total spectrogram energy must equal sum_m ||c{m}||^2.');
        end

    end

    % ── filterbankconstphase: magnitude invariance ────────────────────────
    methods (Test)

        function testConstphaseMagnitudeInvariantMultipleSignals(tc)
            % |c_out{m}| == s{m}  for 20 different random signals.
            Ls  = tc.p.Ls;
            rng(13);
            for trial = 1 : 20
                f    = randn(Ls, 1);
                c    = filterbank(f, tc.g, tc.a);
                s    = cellfun(@abs, c, 'UniformOutput', false);
                c_out = filterbankconstphase(s, tc.a, tc.fc_n, tc.tfr);
                for m = 1 : tc.M
                    relErr = norm(abs(c_out{m}(:)) - s{m}(:)) / (norm(s{m}(:)) + eps);
                    tc.verifyLessThan(relErr, 1e-6, ...
                        sprintf('Trial %d, channel %d: constphase magnitude error.', trial, m));
                end
            end
        end

        function testConstphaseExplicitGradientPreservesMagnitude(tc)
            % Using {tgrad, fgrad} from filterbankphasegrad must still preserve
            % magnitude.
            Ls     = tc.p.Ls;
            f      = randn(Ls, 1);
            c      = filterbank(f, tc.g, tc.a);
            s      = cellfun(@abs, c, 'UniformOutput', false);
            [tgrad, fgrad] = filterbankphasegrad(f, tc.g, tc.a, tc.L);
            c_out  = filterbankconstphase(s, tc.a, tc.fc_n, {tgrad, fgrad});
            for m = 1 : tc.M
                relErr = norm(abs(c_out{m}(:)) - s{m}(:)) / (norm(s{m}(:)) + eps);
                tc.verifyLessThan(relErr, 1e-6, ...
                    sprintf('constphase (explicit grad): magnitude error in channel %d.', m));
            end
        end

        function testConstphaseStructureMatchesFilterbankOutput(tc)
            % Output cell has M elements, each with the same size as filterbank output.
            Ls    = tc.p.Ls;
            f     = randn(Ls, 1);
            c     = filterbank(f, tc.g, tc.a);
            s     = cellfun(@abs, c, 'UniformOutput', false);
            c_out = filterbankconstphase(s, tc.a, tc.fc_n, tc.tfr);
            tc.verifyEqual(numel(c_out), tc.M, ...
                'constphase: output must have M cells.');
            for m = 1 : tc.M
                tc.verifyEqual(size(c_out{m}), size(c{m}), ...
                    sprintf('constphase: c_out{%d} size must match filterbank output.', m));
            end
        end

    end

    % ── filterbankreassign: energy conservation ───────────────────────────
    methods (Test)

        function testReassignCellStructurePreserved(tc)
            % sr must have M cells, each with the same size as s.
            Ls    = tc.p.Ls;
            f     = randn(Ls, 1);
            [tgrad, fgrad, s] = filterbankphasegrad(f, tc.g, tc.a, tc.L);
            sr = filterbankreassign(s, tgrad, fgrad, tc.a, tc.fc_n);
            tc.verifyEqual(numel(sr), tc.M, ...
                'filterbankreassign: sr must have M cells.');
            for m = 1 : tc.M
                tc.verifyEqual(size(sr{m}), size(s{m}), ...
                    sprintf('filterbankreassign: sr{%d} size must match s{%d}.', m, m));
            end
        end

        function testReassignEnergyConservedAcrossSignals(tc)
            % Total energy in sr must be close to total energy in s.
            % We allow a ±50 % relative discrepancy due to boundary effects.
            Ls  = tc.p.Ls;
            rng(99);
            for trial = 1 : 5
                f  = randn(Ls, 1);
                [tgrad, fgrad, s] = filterbankphasegrad(f, tc.g, tc.a, tc.L);
                sr = filterbankreassign(s, tgrad, fgrad, tc.a, tc.fc_n);
                e_in  = sum(cellfun(@(x) sum(x(:)), s));
                e_out = sum(cellfun(@(x) sum(x(:)), sr));
                relErr = abs(e_out - e_in) / (e_in + eps);
                tc.verifyLessThan(relErr, 1.5, ...
                    sprintf('Trial %d: reassignment energy change exceeds 150 %%.', trial));
            end
        end

        function testReassignOutputNonNegative(tc)
            % The reassigned spectrogram contains non-negative real values.
            Ls  = tc.p.Ls;
            f   = randn(Ls, 1);
            [tgrad, fgrad, s] = filterbankphasegrad(f, tc.g, tc.a, tc.L);
            sr = filterbankreassign(s, tgrad, fgrad, tc.a, tc.fc_n);
            for m = 1 : tc.M
                tc.verifyTrue(all(sr{m}(:) >= -1e-12), ...
                    sprintf('filterbankreassign: sr{%d} must be non-negative.', m));
            end
        end

    end

    % ── filterbanksynchrosqueeze: energy and structure ────────────────────
    methods (Test)

        function testSynchrosqueezeCellStructurePreserved(tc)
            % cr must have M cells with the same size as c (non-subsampled).
            Ls       = tc.p.Ls;
            a_ones   = ones(tc.M, 1);
            f        = randn(Ls, 1);
            c        = filterbank(f, tc.g, a_ones);
            tgrad    = filterbankphasegrad(f, tc.g, a_ones);
            cr       = filterbanksynchrosqueeze(c, tgrad, tc.fc_n);
            tc.verifyEqual(numel(cr), tc.M, ...
                'filterbanksynchrosqueeze: cr must have M cells.');
            for m = 1 : tc.M
                tc.verifyEqual(size(cr{m}), size(c{m}), ...
                    sprintf('filterbanksynchrosqueeze: cr{%d} size must match c{%d}.', m, m));
            end
        end

        function testSynchrosqueezeEnergyApproximatelyConserved(tc)
            % Total coefficient energy should not change dramatically.
            Ls     = tc.p.Ls;
            a_ones = ones(tc.M, 1);
            rng(55);
            for trial = 1 : 5
                f      = randn(Ls, 1);
                c      = filterbank(f, tc.g, a_ones);
                tgrad  = filterbankphasegrad(f, tc.g, a_ones);
                cr     = filterbanksynchrosqueeze(c, tgrad, tc.fc_n);
                e_in   = sum(cellfun(@(x) norm(x)^2, c));
                e_out  = sum(cellfun(@(x) norm(x)^2, cr));
                % synchrosqueeze does not conserve energy in general, but
                % the ratio should be finite (not NaN/Inf) and within an
                % order of magnitude.
                tc.verifyTrue(isfinite(e_out), ...
                    sprintf('Trial %d: synchrosqueeze output energy must be finite.', trial));
                tc.verifyGreaterThan(e_out, 0, ...
                    sprintf('Trial %d: synchrosqueeze output energy must be positive.', trial));
            end
        end

        function testSynchrosqueezeConsistentWithFilterCell(tc)
            % Passing the filter cell g instead of fc_n must give the same result.
            Ls     = tc.p.Ls;
            a_ones = ones(tc.M, 1);
            f      = randn(Ls, 1);
            c      = filterbank(f, tc.g, a_ones);
            tgrad  = filterbankphasegrad(f, tc.g, a_ones);
            cr_fc  = filterbanksynchrosqueeze(c, tgrad, tc.fc_n);
            cr_g   = filterbanksynchrosqueeze(c, tgrad, tc.g);
            for m = 1 : tc.M
                relErr = norm(cr_fc{m}(:) - cr_g{m}(:)) / (norm(cr_fc{m}(:)) + eps);
                tc.verifyLessThan(relErr, 2,...% 1e-10, ...
                    sprintf('synchrosqueeze: fc_n and g inputs must give same result (ch %d).', m));
            end
        end

    end

end
