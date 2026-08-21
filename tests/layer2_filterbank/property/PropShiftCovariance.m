classdef PropShiftCovariance < matlab.unittest.TestCase
    % PropShiftCovariance - Property test for shift covariance
    %
    % For a filterbank with hop sizes a(m), applying a covariant shift of
    % k*a(m) samples in the input shifts subband m by exactly k samples:
    %
    %   filterbank(circshift(x, k*a(m)), g, a){m}
    %       == circshift(filterbank(x, g, a){m}, k)
    %
    % IMPORTANT: signals must have length L (the DFT length returned by
    % audfilters), NOT Ls.  The filterbank is implemented as circular
    % convolution with period L; the shift-covariance theorem requires the
    % input to be periodic with period L.  Using signals of length Ls < L
    % would introduce a periodicity mismatch and break the identity.

    properties
        p   % Parameters: fs, Ls
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

        function testShiftCovarianceRealSignals(tc)
            % Shift covariance with real and complex-valued signals of length L.
            [g, a, ~, L] = audfilters(tc.p.fs, tc.p.Ls);
            M            = length(a);

            % Representative subbands: first, middle, last
            subband_indices = [1, ceil(M/2), M];

            rng(1);
            for sig_idx = 1:2
                if sig_idx == 1
                    x = randn(L, 1);           % real signal of length L
                else
                    x = randn(L, 1) + 1i*randn(L, 1);  % complex signal of length L
                end

                c_orig = filterbank(x, g, a);

                for k = 1:3
                    for m_idx = 1:length(subband_indices)
                        m             = subband_indices(m_idx);
                        shift_amount  = k * a(m, 1);
                        x_shifted     = circshift(x, shift_amount);

                        c_shift = filterbank(x_shifted, g, a);

                        c_m_expected = circshift(c_orig{m}, k);
                        relError     = norm(c_shift{m} - c_m_expected) / ...
                                       (norm(c_m_expected) + 1e-10);

                        tc.verifyLessThan(relError, 1e-6, ...
                            sprintf('sig%d, shift k=%d, subband m=%d: error %.2e', ...
                            sig_idx, k, m, relError));
                    end
                end
            end
        end

        function testShiftCovarianceRandomSignals(tc)
            % Shift covariance with 100 random complex signals of length L.
            [g, a, ~, L] = audfilters(tc.p.fs, tc.p.Ls);
            M            = length(a);
            m            = 1;   % test first subband (representative)

            for trial = 1:100
                x      = randn(L, 1) + 1i*randn(L, 1);
                c_orig = filterbank(x, g, a);

                k            = 1;
                shift_amount = k * a(m, 1);
                x_shifted    = circshift(x, shift_amount);
                c_shift      = filterbank(x_shifted, g, a);

                c_m_expected = circshift(c_orig{m}, k);
                relError     = norm(c_shift{m} - c_m_expected) / ...
                               (norm(c_m_expected) + 1e-10);

                tc.verifyLessThan(relError, 1e-6, ...
                    sprintf('Trial %d: shift covariance error %.2e', trial, relError));
            end
        end

        function testShiftCovarianceMultipleShifts(tc)
            % Shift covariance for k = 1, 2, 3 on every subband.
            [g, a, ~, L] = audfilters(tc.p.fs, tc.p.Ls);
            M            = length(a);

            rng(42);
            x      = randn(L, 1) + 1i*randn(L, 1);
            c_orig = filterbank(x, g, a);

            for k = 1:3
                for m = 1:M
                    shift_amount = k * a(m, 1);
                    x_shifted    = circshift(x, shift_amount);
                    c_shift      = filterbank(x_shifted, g, a);

                    c_m_expected = circshift(c_orig{m}, k);
                    relError     = norm(c_shift{m} - c_m_expected) / ...
                                   (norm(c_m_expected) + 1e-10);

                    tc.verifyLessThan(relError, 1e-6, ...
                        sprintf('shift k=%d, subband m=%d: error %.2e', k, m, relError));
                end
            end
        end

    end
end
