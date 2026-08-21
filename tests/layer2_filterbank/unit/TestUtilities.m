classdef TestUtilities < matlab.unittest.TestCase
%TESTUTILITIES  Unit tests for utility entry points.
%
%   Covers: nonu2ucfmt, u2nonucfmt, filterbanklengthcoef, cent_freqs.

    properties
        sig
        p
        g       % ERB filters
        a       % subsampling
        fc      % center frequencies (Hz)
        L       % system length
        M       % filter count
        % nonu2u conversion data (built once)
        gu      % uniform equivalent filters
        au      % uniform hop
        p_vec   % copies-per-channel vector
        c       % non-uniform coefficients from filterbank
        cu      % uniform coefficients from nonu2ucfmt
    end

    methods (TestClassSetup)
        function setupClass(tc)
            sharedDir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'shared');
            addpath(sharedDir);
            setup_filterbank_paths();
            [tc.sig, tc.p] = make_test_params();

            [tc.g, tc.a, tc.fc, tc.L] = audfilters(tc.p.fs, tc.p.Ls);
            tc.M = numel(tc.g);

            % Build uniform equivalent bank and coefficient format.
            [tc.gu, tc.au, tc.p_vec] = nonu2ufilterbank(tc.g, tc.a);
            tc.c  = filterbank(tc.sig.noise_mono, tc.g, tc.a);
            tc.cu = nonu2ucfmt(tc.c, tc.p_vec);
        end
    end

    % ── nonu2ucfmt ────────────────────────────────────────────────────────
    methods (Test)

        function testNonu2ucfmtOutputIsCell(tc)
            tc.verifyTrue(iscell(tc.cu), ...
                'nonu2ucfmt: output must be a cell array.');
        end

        function testNonu2ucfmtOutputLength(tc)
            tc.verifyEqual(numel(tc.cu), sum(tc.p_vec), ...
                'nonu2ucfmt: output must have sum(p) cells.');
        end

        function testNonu2ucfmtSubchannelColumns(tc)
            % Each uniform sub-channel must have 1 column (mono input).
            for k = 1 : numel(tc.cu)
                tc.verifyEqual(size(tc.cu{k}, 2), 1, ...
                    sprintf('nonu2ucfmt: cu{%d} must have 1 column for mono input.', k));
            end
        end

    end

    % ── u2nonucfmt ────────────────────────────────────────────────────────
    methods (Test)

        function testU2nonucfmtRoundtrip(tc)
            % nonu2ucfmt followed by u2nonucfmt must recover the original.
            c2 = u2nonucfmt(tc.cu, tc.p_vec);
            tc.verifyEqual(numel(c2), tc.M, ...
                'u2nonucfmt roundtrip: output must have M channels.');
            err = zeros(1, tc.M);
            for m = 1 : tc.M
                denom = norm(tc.c{m}, 'fro') + eps;
                err(m) = norm(c2{m} - tc.c{m}, 'fro') / denom;
            end
            tc.verifyLessThan(max(err), 1e-10, ...
                'u2nonucfmt(nonu2ucfmt(c, p), p) must recover c exactly.');
        end

        function testU2nonucfmtOutputIsCell(tc)
            c2 = u2nonucfmt(tc.cu, tc.p_vec);
            tc.verifyTrue(iscell(c2), ...
                'u2nonucfmt: output must be a cell array.');
        end

        function testU2nonucfmtOutputLength(tc)
            c2 = u2nonucfmt(tc.cu, tc.p_vec);
            tc.verifyEqual(numel(c2), tc.M, ...
                'u2nonucfmt: output must have M channels (= numel(p)).');
        end

    end

    % ── filterbanklengthcoef ──────────────────────────────────────────────
    methods (Test)

        function testFilterbanklengthcoefMatchesFilterbanklength(tc)
            % The length inferred from the coefficients must equal
            % the length reported by filterbanklength.
            L_from_coef = filterbanklengthcoef(tc.c, tc.a);
            tc.verifyEqual(L_from_coef, tc.L, ...
                'filterbanklengthcoef: must return the same L as filterbanklength.');
        end

        function testFilterbanklengthcoefIsPositiveInteger(tc)
            L_fc = filterbanklengthcoef(tc.c, tc.a);
            tc.verifyGreaterThan(L_fc, 0, ...
                'filterbanklengthcoef: output must be positive.');
            tc.verifyEqual(L_fc, round(L_fc), ...
                'filterbanklengthcoef: output must be an integer.');
        end

        function testFilterbanklengthcoefStereoConsistent(tc)
            % Stereo input should give the same length as mono.
            c_stereo = filterbank(tc.sig.noise_stereo, tc.g, tc.a);
            L_stereo = filterbanklengthcoef(c_stereo, tc.a);
            tc.verifyEqual(L_stereo, tc.L, ...
                'filterbanklengthcoef: stereo input must give the same L as mono.');
        end

    end

    % ── cent_freqs ────────────────────────────────────────────────────────
    methods (Test)

        function testCentFreqsLength(tc)
            cfreq = cent_freqs(tc.g, tc.L);
            tc.verifyEqual(numel(cfreq), tc.M, ...
                'cent_freqs: output length must equal number of filters.');
        end

        function testCentFreqsInNormalisedRange(tc)
            cfreq = cent_freqs(tc.g, tc.L);
            tc.verifyTrue(all(abs(cfreq) <= 1 + 1e-12), ...
                'cent_freqs: all values must lie in (-1, 1].');
        end

        function testCentFreqsMonotonic(tc)
            % For the ERB filter bank, center frequencies should be ordered.
            % The last (highpass) boundary filter is excluded: its circular centre
            % of gravity wraps past the Nyquist bin and maps to a negative value,
            % which is a known artefact of the periodic DFT, not an ordering bug.
            % (testCentFreqsMatchAudfiltersNormalized already verifies the interior
            % filters are consistent with audfilters fc, using the same exclusion.)
            cfreq = cent_freqs(tc.g, tc.L);
            cfreq_interior = cfreq(1 : tc.M - 1);
            if numel(cfreq_interior) >= 2
                tc.verifyTrue(all(diff(cfreq_interior) > -1e-10), ...
                    'cent_freqs: center frequencies must be non-decreasing (excluding last boundary filter).');
            end
        end

        function testCentFreqsMatchAudfiltersNormalized(tc)
            % Normalised cent_freqs should equal fc_Hz / (fs/2).
            cfreq      = cent_freqs(tc.g, tc.L);
            fc_norm    = tc.fc / (tc.p.fs / 2);
            % The two frequency vectors should be consistent (correlation > 0.99).
            % We compare the interior filters only, avoiding DC / Nyquist edge filters.
            interior   = 2 : tc.M - 1;
            if numel(interior) >= 2
                r = corrcoef(cfreq(interior), fc_norm(interior));
                tc.verifyGreaterThan(r(1,2), 0.99, ...
                    'cent_freqs: must be consistent with audfilters fc (correlation check).');
            end
        end

    end

end
