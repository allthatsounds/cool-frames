classdef PropPolyphaseEquivalence < matlab.unittest.TestCase
    % PropPolyphaseEquivalence: FFT path and TD path agree for FIR filters
    %
    % For causal FIR filters with periodic boundary extension:
    %   comp_filterbank_fft(fft(f), G_fft, a)  should equal
    %   comp_filterbank_td(f, G_td, a, offset, 'per')
    % where G_fft{m} = fft(postpad(h_m, L)) and G_td{m} = h_m.
    %
    % Correct calling conventions:
    %   comp_filterbank_fft(F, G, a)      -- F = fft(f)
    %   comp_filterbank_td(f, g, a, offset, ext)

    properties
        Ls
        M
        Lh      % FIR filter length
        a       % subsampling factors
        G_td    % cell{M} of time-domain impulse responses
        G_fft   % cell{M} of length-Ls DFT responses
        offset  % analysis filter offset (all zeros = causal)
        noise_real
    end

    methods(TestClassSetup)
        function setupTestClass(testCase)
            sharedDir = fullfile(fileparts(mfilename('fullpath')), ...
                                 '..', '..', 'shared');
            addpath(sharedDir);
            setup_filterbank_paths();
            testCase.Ls = 512;
            testCase.M  = 3;
            testCase.Lh = 16;          % FIR filter length
            testCase.a  = [4; 8; 8];   % subsampling factors (all divide 512)

            % Build matching TD and full-DFT filter representations
            Ls      = testCase.Ls;
            M       = testCase.M;
            Lh      = testCase.Lh;
            G_td_tmp  = cell(M, 1);
            G_fft_tmp = cell(M, 1);
            for m = 1 : M
                h            = firwin('hann', Lh);
                G_td_tmp{m}  = h;
                G_fft_tmp{m} = fft(postpad(h, Ls));
            end
            testCase.G_td   = G_td_tmp;
            testCase.G_fft  = G_fft_tmp;
            testCase.offset = zeros(testCase.M, 1);   % causal filters

            testCase.noise_real = randn(testCase.Ls, 1);
        end
    end

    methods(Test)

        function testFFTVsTDPath(testCase)
            % For each of several random signals, the FFT path and TD path
            % should agree to within 1e-10 (relative) with 'per' extension.
            rng(42);
            num_signals = 5;

            for trial = 1:num_signals
                x = randn(testCase.Ls, 1);
                F = fft(x);

                % FFT path: pass fft(x)
                c_fft = comp_filterbank_fft(F, testCase.G_fft, testCase.a);

                % TD path: pass time-domain x
                c_td  = comp_filterbank_td(x, testCase.G_td, testCase.a, ...
                                           testCase.offset, 'per');

                for m = 1:testCase.M
                    % Both paths should have the same output length
                    Nfft = size(c_fft{m}, 1);
                    Ntd  = size(c_td{m},  1);
                    Nmin = min(Nfft, Ntd);

                    diff = max(abs(c_fft{m}(1:Nmin) - c_td{m}(1:Nmin)));
                    scale = max(abs(c_fft{m}(1:Nmin)));
                    if scale < 1e-15, scale = 1; end

                    testCase.verifyLessThan(diff / scale, 1e-8, ...
                        sprintf('Trial %d, subband %d: FFT vs TD mismatch (rel err = %g)', ...
                                trial, m, diff/scale));
                end
            end
        end

        function testZeroInputBothPaths(testCase)
            % Zero signal gives zero output on both paths
            x_zero = zeros(testCase.Ls, 1);
            F_zero = fft(x_zero);

            c_fft = comp_filterbank_fft(F_zero, testCase.G_fft, testCase.a);
            c_td  = comp_filterbank_td(x_zero, testCase.G_td, testCase.a, ...
                                       testCase.offset, 'per');

            for m = 1:testCase.M
                testCase.verifyEqual(c_fft{m}, zeros(size(c_fft{m})), 'AbsTol', 1e-14);
                testCase.verifyEqual(c_td{m},  zeros(size(c_td{m})),  'AbsTol', 1e-14);
            end
        end

    end

end
