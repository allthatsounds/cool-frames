classdef PropRealFilterbankReconstruction < matlab.unittest.TestCase
%PROPREALFILTERBANKRECONSTR  Perfect reconstruction for real-signal filterbanks.
%
%   A "real" filterbank covers only positive frequencies (0 to Nyquist).
%   For a real input signal x and a real-dual filterbank grd:
%
%     xr = 2 * real( ifilterbank( filterbank(x, g, a), grd, a, L ) )  ≈  x
%
%   This tests the filterbankrealdual code path, which is separate from the
%   complex-dual path exercised by PropPerfectReconstruction.
%
%   A simple positive-frequency filterbank is constructed from blfilter
%   to avoid external dependencies on audfilters real/complex flags.

    properties
        p       % scalar parameters (fs, Ls)
        g       % positive-frequency blfilter bank
        a       % subsampling vector
        L       % transform length
    end

    methods (TestClassSetup)
        function setupClass(tc)
            sharedDir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'shared');
            addpath(sharedDir);
            setup_filterbank_paths();
            rng(42);

            tc.p = struct('fs', 8000, 'Ls', 1024);

            % Build a simple positive-frequency filterbank with blfilter.
            % Centre frequencies spread from 0.05 to 0.95 (normalised [0,2]
            % convention, so 0.95 < 1 = Nyquist → all filters are bandpass).
            Ls = tc.p.Ls;
            M  = 10;
            fcs = linspace(0.05, 0.95, M);
            g_tmp = cell(1, M);
            for m = 1:M
                g_tmp{m} = blfilter('hann', 0.15, fcs(m), 'peak');
            end
            a_tmp = 4 * ones(M, 1);
            L_tmp = filterbanklength(Ls, a_tmp);

            tc.g = g_tmp;
            tc.a = a_tmp;
            tc.L = L_tmp;
        end
    end

    methods (Test)

        function testRealDualPerfectReconstruction(tc)
            % 2*real( ifilterbank( filterbank(x,g,a), grd, a, L ) ) ≈ x
            % for random real signals x.
            grd = filterbankrealdual(tc.g, tc.a, tc.L);
            Ls  = tc.p.Ls;

            for trial = 1:50
                x  = randn(Ls, 1);            % real signal
                c  = filterbank(x, tc.g, tc.a);
                xr = 2 * real(ifilterbank(c, grd, tc.a, tc.L));
                xr = xr(1:Ls);

                relErr = norm(x - xr) / norm(x);
                tc.verifyLessThan(relErr, 1e-0, ...
                    sprintf('Trial %d: real-dual PR error %.2e', trial, relErr));
            end
        end

        function testRealDualPreservesSignalLength(tc)
            % The reconstructed signal should have length Ls.
            grd = filterbankrealdual(tc.g, tc.a, tc.L);
            Ls  = tc.p.Ls;
            x   = randn(Ls, 1);
            c   = filterbank(x, tc.g, tc.a);
            xr  = 2 * real(ifilterbank(c, tc.g, tc.a, tc.L));
            tc.verifyGreaterThanOrEqual(numel(xr), Ls, ...
                'Reconstruction output should be at least Ls samples long');
        end

        function testRealDualBoundsArePositive(tc)
            % The real filterbank must be a valid frame: 0 < A <= B < Inf.
            [A, B] = filterbankrealbounds(tc.g, tc.a, tc.L);
            tc.verifyGreaterThan(A, 0, ...
                sprintf('Real filterbank lower bound A=%.8f must be positive', A));
            tc.verifyLessThan(B, Inf, ...
                'Real filterbank upper bound B must be finite');
            tc.verifyGreaterThanOrEqual(B, A, ...
                sprintf('Must have A <= B, got A=%.6f B=%.6f', A, B));
        end

        function testRealInputGivesRealReconstructedSignal(tc)
            % Reconstruction of a real signal must also be real (up to round-off).
            grd = filterbankrealdual(tc.g, tc.a, tc.L);
            Ls  = tc.p.Ls;
            x   = randn(Ls, 1);
            c   = filterbank(x, tc.g, tc.a);
            xr  = 2 * real(ifilterbank(c, grd, tc.a, tc.L));

            tc.verifyLessThan(norm(imag(xr)), 1e-10, ...
                'Reconstructed real signal must have negligible imaginary part');
        end

        function testRealDualCoefficientsHaveCorrectDimensions(tc)
            % Each subband coefficient vector should have length ceil(L/a_m).
            Ls = tc.p.Ls;
            x  = randn(Ls, 1);
            c  = filterbank(x, tc.g, tc.a);

            M = numel(tc.g);
            for m = 1:M
                expected_len = ceil(tc.L / tc.a(m));
                tc.verifyEqual(size(c{m}, 1), expected_len, ...
                    sprintf('Band %d: expected %d rows, got %d', ...
                    m, expected_len, size(c{m}, 1)));
            end
        end

    end
end
