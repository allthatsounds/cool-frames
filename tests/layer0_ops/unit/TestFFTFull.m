classdef TestFFTFull < matlab.unittest.TestCase
    % TestFFTFull: unit tests for comp_filterbank_fft and comp_ifilterbank_fft
    %
    % Correct calling conventions (from comp_filterbank.m):
    %   c = comp_filterbank_fft(F, G, a)
    %       F : fft(f), size [L x W]          -- FFT of the signal, NOT the signal itself
    %       G : cell{M} of length-L DFT responses
    %       a : integer subsampling vector, a(m) must divide L exactly
    %
    %   F = comp_ifilterbank_fft(c, G, a)
    %       c : cell{M} of coefficient matrices [N(m) x W]
    %       G : same cell of length-L DFT responses
    %       a : same subsampling vector
    %       (no Ls argument -- output length = numel(G{1}))

    properties
        Ls           % signal length
        M            % number of subbands
        a            % subsampling factors  [M x 1]
        G            % cell{M} of rectangular bandpass DFT responses (length Ls)
        noise_real
        noise_stereo
        zeros_sig
    end

    methods(TestClassSetup)
        function setupTestClass(testCase)
            sharedDir = fullfile(fileparts(mfilename('fullpath')), ...
                                 '..', '..', 'shared');
            addpath(sharedDir);
            setup_filterbank_paths();
            testCase.Ls = 1024;
            testCase.M  = 4;
            testCase.a  = [4; 8; 8; 16];   % subsampling factors (all divide 1024)

            % Build M rectangular full-length DFT responses
            Ls    = testCase.Ls;
            M     = testCase.M;
            bw    = floor(Ls / (2 * M + 2));   % bandwidth per band (in bins)
            G_tmp = cell(M, 1);
            for m = 1 : M
                g_full  = zeros(Ls, 1);
                fc_bin  = round(m * Ls / (2 * (M + 1)));
                lo = max(1, fc_bin - floor(bw/2));
                hi = min(Ls, fc_bin + floor(bw/2));
                g_full(lo : hi) = 1;
                G_tmp{m} = g_full;
            end
            testCase.G = G_tmp;

            testCase.noise_real   = randn(testCase.Ls, 1);
            testCase.noise_stereo = randn(testCase.Ls, 2);
            testCase.zeros_sig    = zeros(testCase.Ls, 1);
        end
    end

    methods(Test)

        function testFFTFilterbankZeroInput(testCase)
            % Zero signal -> all subbands zero
            F = fft(testCase.zeros_sig);
            c = comp_filterbank_fft(F, testCase.G, testCase.a);

            for m = 1:testCase.M
                testCase.verifyEqual(c{m}, zeros(size(c{m})), 'AbsTol', 1e-14, ...
                    sprintf('Subband %d not zero for zero input', m));
            end
        end

        function testFFTFilterbankOutputSizes(testCase)
            % Each c{m} has N(m) = Ls/a(m) rows
            F = fft(testCase.noise_real);
            c = comp_filterbank_fft(F, testCase.G, testCase.a);

            for m = 1:testCase.M
                expected = testCase.Ls / testCase.a(m);
                testCase.verifyEqual(size(c{m}, 1), expected, ...
                    sprintf('Subband %d: expected %d rows, got %d', m, expected, size(c{m},1)));
            end
        end

        function testFFTFilterbankRealNoise(testCase)
            % Real noise: M subbands returned
            F = fft(testCase.noise_real);
            c = comp_filterbank_fft(F, testCase.G, testCase.a);
            testCase.verifyEqual(length(c), testCase.M);
        end

        function testFFTFilterbankMultiChannel(testCase)
            % Multi-channel [Ls x 2]: each c{m} has 2 columns
            F = fft(testCase.noise_stereo);
            c = comp_filterbank_fft(F, testCase.G, testCase.a);

            for m = 1:testCase.M
                testCase.verifyEqual(size(c{m}, 2), 2, ...
                    sprintf('Subband %d: expected 2 columns, got %d', m, size(c{m},2)));
            end
        end

        function testFFTFilterbankIntegerSubsampling(testCase)
            % All a(m) divide Ls exactly (required by comp_filterbank_fft)
            testCase.verifyTrue(all(mod(testCase.Ls, testCase.a) == 0));
        end

        function testInverseFFTFilterbankZeroInput(testCase)
            % Zero coefficients -> zero output from synthesis.
            % comp_ifilterbank_fft requires numel(c{m}) = Ls/a(m) so that
            % repmat(fft(c{m}), a(m), 1) tiles to exactly Ls samples.
            % Each channel therefore needs its own length Ls/a(m).
            c_zero = cell(testCase.M, 1);
            for m = 1:testCase.M
                N_m = testCase.Ls / testCase.a(m);
                c_zero{m} = zeros(N_m, 1);
            end
            F_recon = comp_ifilterbank_fft(c_zero, testCase.G, testCase.a);
            testCase.verifyEqual(F_recon, zeros(testCase.Ls, 1), 'AbsTol', 1e-10);
        end

        function testForwardInverseRoundtrip(testCase)
            % For the trivial M=1 all-pass filter (G{1} = ones(L,1), a=1):
            %   comp_filterbank_fft(fft(f), G, a)    -> c{1} = f
            %   comp_ifilterbank_fft(c, G, a)        -> F_recon = fft(f)
            % so ifft(F_recon) = f exactly.
            G_trivial = {ones(testCase.Ls, 1)};
            a_trivial = ones(1, 1);   % subsampling = 1
            F       = fft(testCase.noise_real);
            c       = comp_filterbank_fft(F, G_trivial, a_trivial);
            F_recon = comp_ifilterbank_fft(c, G_trivial, a_trivial);
            f_recon = real(ifft(F_recon));

            err = norm(f_recon - testCase.noise_real) / norm(testCase.noise_real);
            testCase.verifyLessThan(err, 1e-10, ...
                'All-pass roundtrip: ifft(F_recon) should equal f');
        end

    end

end
